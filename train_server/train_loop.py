# -*- coding:utf-8 -*-

import json
import logging
import numpy as np
from collections import defaultdict
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import dist_train_config
import dist_params
import varserverapi
from dataset_loader import DatasetLoader
from weight_pack import WeightPack
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

train_manager = None
class TrainManager():
    def __init__(self, initial_weight_path, weight_save_path_tmpl, initial_t=0):
        self.clients = []
        self.t = initial_t
        self.weight_save_path_tmpl = weight_save_path_tmpl
        self.dataset_loader = DatasetLoader.init_by_setting(dist_params)
        self.varserver_db = varserverapi.open_db()

        self.reserved_vars_by_t = defaultdict(list)#variable ids used for specific iteration
        packer = WeightPack(json.load(open(dist_params.WEIGHT_PACK_PARAM_FILE)))
        initial_weights = packer.unpack(open(initial_weight_path, "rb").read())
        self.optimizer = Optimizer(initial_weights, packer, lr=dist_params.LR)
        self.weight_id = None
        self.gradient_ids = []
        self.n_gradient_gathered = 0
    
    def clientConnected(self, client):
        client.client_id = len(self.clients)
        self.clients.append(client)
        logging.info("Client {} connected ({})".format(client.client_id, client.address[0]))
        if len(self.clients) == dist_params.N_CLIENTS:
            #all clients connected; start training
            self.start_training()
    
    def clientClose(self, client):
        self.clients.remove(client)
        logging.warn("Client {} disconnected ({})".format(client.client_id, client.address[0]))
    
    def clientMessage(self, client):
        command = json.loads(client.data)
        logging.info("Client {} command {}".format(client.client_id, client.data))
        if command["command"] == "stored_gradient":
            self.n_gradient_gathered += 1
            self.gradient_multipliers[command["gradient_id"]] = command["gradient_multiplier"]
            self.gradient_batch_sizes[command["gradient_id"]] = command["batch_size"]
            if self.n_gradient_gathered == dist_params.N_CLIENTS:
                # all gradients gathered
                # calculate new weight
                logging.info("Updating weight by gradients {}".format(self.gradient_ids))
                self.optimizer.update_weight(self.varserver_db, self.gradient_ids, self.gradient_multipliers, self.gradient_batch_sizes)

                #remove variables
                for var_id in self.reserved_vars_by_t[self.t]:
                    varserverapi.remove_var(self.varserver_db, var_id)
                del self.reserved_vars_by_t[self.t]

                # start new iteration
                self.t += 1
                self.start_iteration()
        elif command["command"] == "loaded_weight":
            # when all clients loaded weights, feed next data
            self.n_client_loaded_weight += 1
            if self.n_client_loaded_weight == dist_params.N_CLIENTS:
                self.feed_data(self.t + 1)
    
    def start_training(self):
        # load initial data and start iteration
        logging.info("Start training")
        self.feed_data(self.t)
        self.start_iteration()

    def start_iteration(self):
        self.n_gradient_gathered = 0
        self.n_client_loaded_weight = 0
        self.gradient_multipliers = {}
        self.gradient_batch_sizes = {}
        logging.info("Stroing current weight")
        self.weight_id = self.optimizer.store_weight(self.varserver_db)
        self.reserved_vars_by_t[self.t].append(self.weight_id)

        if self.t % dist_params.MODEL_SAVE_ITER == 0:
            self.save_model()
        logging.info("Start iteration {} weight {}".format(self.t, self.weight_id))
        self.gradient_ids = []
        for client_id in range(dist_params.N_CLIENTS):
            client = self.clients[client_id]
            gradient_id = varserverapi.reserve_var(self.varserver_db)
            self.gradient_ids.append(gradient_id)
            self.reserved_vars_by_t[self.t].append(gradient_id)
            client.sendMessage(unicode(json.dumps({"command":"calc_gradient", "vars":{"weight":self.weight_id, "gradient":gradient_id}})))
        #self.feed_data(self.t + 1)#pre-fetch
    
    def save_model(self):
        try:
            logging.info("Saving model at {}".format(self.t))
            save_weight_id = self.optimizer.store_weight(self.varserver_db)
            weight_blob = varserverapi.read_var(self.varserver_db, save_weight_id)
            with open(self.weight_save_path_tmpl % self.t, "wb") as f:
                f.write(weight_blob)
            varserverapi.remove_var(self.varserver_db, save_weight_id)
        except Exception as ex:
            print(ex)
    
    def feed_data(self, t):
        """
        Load iteration training data from dataset and place on variable server 
        Request clients to fetch data for specified iteration
        """
        logging.info("Feeding data {}".format(t))
        offset_t = t * dist_params.BATCH_SIZE
        for client_id in range(dist_params.N_CLIENTS):
            try:
                data_bin, label_bin = self.dataset_loader.load(offset_t + client_id * dist_params.CLIENT_BATCH_SIZE, dist_params.CLIENT_BATCH_SIZE)
                client = self.clients[client_id]
                data_id = varserverapi.reserve_write_var(self.varserver_db, data_bin)
                label_id = varserverapi.reserve_write_var(self.varserver_db, label_bin)
                self.reserved_vars_by_t[t].append(data_id)
                self.reserved_vars_by_t[t].append(label_id)
                logging.info(unicode(json.dumps({"command":"read_data", "vars":{"data":data_id, "label":label_id}})))
            except Exception as ex:
                logging.error(str(ex))
            client.sendMessage(unicode(json.dumps({"command":"read_data", "vars":{"data":data_id, "label":label_id}})))

class Optimizer():
    def __init__(self, weights, packer, lr=1e-2, momentum=0.9):
        self.weights = weights
        self.packer = packer
        self.lr = lr
        self.momentum = momentum
        self.last_delta = {name:np.zeros_like(w) for name, w in weights.items()}
    
    def store_weight(self, varserver_db):
        return varserverapi.reserve_write_var(varserver_db, self.packer.pack(self.weights))
    
    def update_weight(self, varserver_db, gradient_ids, gradient_multipliers, batch_sizes):
        gradients = {}
        total_batch_size = 0
        # sum up all gradients
        for gradient_id in gradient_ids:
            logging.debug("reading gradient")
            gradient_bin = varserverapi.read_var(varserver_db, gradient_id)
            logging.debug("unpacking gradient")
            client_gradients = self.packer.unpack(gradient_bin)
            batch_size = batch_sizes[gradient_id]
            gradient_multiplier = gradient_multipliers[gradient_id]
            total_batch_size += batch_size
            logging.debug("accumlating gradient")
            for name, cg in client_gradients.items():
                #logging.info("sum: {}, {}".format(np.sum(cg), name))
                cg *= (float(batch_size) / gradient_multiplier)
                if name in gradients:
                    gradients[name] += cg
                else:
                    gradients[name] = cg
        # do update
        logging.debug("updating weight")
        for name, g in gradients.items():
            delta = self.last_delta[name]
            delta *= self.momentum
            delta += g * (-self.lr / total_batch_size)
            self.weights[name] += delta

class ClientConnection(WebSocket):
    def __init__(self, *args, **kwargs):
        super(ClientConnection, self).__init__(*args, **kwargs)
        self.client_id = None#0, 1, ...

    def handleMessage(self):
        train_manager.clientMessage(self)

    def handleConnected(self):
        train_manager.clientConnected(self)

    def handleClose(self):
        train_manager.clientClose(self)

def train_loop(**tmgr_params):
    global train_manager
    train_manager = TrainManager(**tmgr_params)
    server = SimpleWebSocketServer(dist_train_config.WS_HTTP_HOST, dist_train_config.WS_HTTP_PORT, ClientConnection)
    server.serveforever()
