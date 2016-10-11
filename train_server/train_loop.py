# -*- coding:utf-8 -*-

import json
import logging
import numpy as np
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import dist_train_config
import dist_params
import varserverapi
from dataset_loader import DatasetLoader
from weight_pack import WeightPack
logging.basicConfig(level=logging.DEBUG)

train_manager = None
class TrainManager():
    def __init__(self):
        self.clients = []
        self.t = 0
        self.dataset_loader = DatasetLoader.init_by_setting()
        self.varserver_db = varserverapi.open_db()

        packer = WeightPack(json.load(open(dist_params.WEIGHT_PACK_PARAM_FILE)))
        initial_weights = packer.unpack(open(dist_params.INITIAL_WEIGHT).read())
        self.optimizer = Optimizer(initial_weights, packer, lr=1e-2)
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
        logging.info("Client {} command {}".format(client.client_id, command["command"]))
        if command["command"] == "stored_gradient":
            self.n_gradient_gathered += 1
            self.gradient_multipliers[command["gradient_id"]] = command["gradient_multiplier"]
            self.gradient_batch_sizes[command["gradient_id"]] = command["batch_size"]
            if self.n_gradient_gathered == dist_params.N_CLIENTS:
                # all gradients gathered
                # calculate new weight
                logging.info("Updating weight by gradients {}".format(self.gradient_ids))
                self.optimizer.update_weight(self.varserver_db, self.gradient_ids, self.gradient_multipliers, self.gradient_batch_sizes)
                self.t += 1
                self.start_iteration()
    
    def start_training(self):
        # load initial data and start iteration
        logging.info("Start training")
        self.feed_data(self.t)
        self.start_iteration()

    def start_iteration(self):
        self.n_gradient_gathered = 0
        self.gradient_multipliers = {}
        self.gradient_batch_sizes = {}
        self.weight_id = self.optimizer.store_weight(self.varserver_db)
        logging.info("Start iteration {} weight {}".format(self.t, self.weight_id))
        self.gradient_ids = []
        for client_id in range(dist_params.N_CLIENTS):
            client = self.clients[client_id]
            gradient_id = varserverapi.reserve_var(self.varserver_db)
            self.gradient_ids.append(gradient_id)
            client.sendMessage(unicode(json.dumps({"command":"calc_gradient", "vars":{"weight":self.weight_id, "gradient":gradient_id}})))
        self.feed_data(self.t + 1)#pre-fetch
    
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
            gradient_bin = varserverapi.read_var(varserver_db, gradient_id)
            client_gradients = self.packer.unpack(gradient_bin)
            batch_size = batch_sizes[gradient_id]
            gradient_multiplier = gradient_multipliers[gradient_id]
            total_batch_size += batch_size
            for name, cg in client_gradients.items():
                logging.info("sum: {}, {}".format(np.sum(cg), name))
                cg *= (float(batch_size) / gradient_multiplier)
                if name in gradients:
                    gradients[name] += cg
                else:
                    gradients[name] = cg
        # do update
        for name, g in gradients.items():
            delta = self.last_delta[name]
            delta *= self.momentum
            delta += g * (-self.lr / total_batch_size)
            self.weights[name] += delta
            logging.info("weight std: {}".format(np.std(self.weights[name])))

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

def train_loop():
    global train_manager
    train_manager = TrainManager()
    server = SimpleWebSocketServer(dist_train_config.WS_HTTP_HOST, dist_train_config.WS_HTTP_PORT, ClientConnection)
    server.serveforever()
