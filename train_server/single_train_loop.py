# -*- coding:utf-8 -*-

import json
import logging
import numpy as np
from collections import defaultdict
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import dist_train_config
import single_params as dist_params
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
        self.weights_bin = open(initial_weight_path, "rb").read()
        self.weight_id = None
        self.gradient_ids = []
    
    def clientConnected(self, client):
        client.client_id = len(self.clients)
        self.clients.append(client)
        logging.info("Client {} connected ({})".format(client.client_id, client.address[0]))
        self.start_training()
    
    def clientClose(self, client):
        self.clients.remove(client)
        logging.warn("Client {} disconnected ({})".format(client.client_id, client.address[0]))
    
    def clientMessage(self, client):
        command = json.loads(client.data)
        logging.info("Client {} command {}".format(client.client_id, client.data))
        if command["command"] == "stored_gradient":
            self.weights_bin = varserverapi.read_var(self.varserver_db, self.gradient_ids[0])
            #remove variables
            self.remove_var_at_t(self.t)

            # start new iteration
            self.t += 1
            self.start_iteration()
        elif command["command"] == "iteration_finished":
            self.remove_var_at_t(self.t)
            self.t += 1
            if dist_params.MODEL_SAVE_ITER - (self.t) % dist_params.MODEL_SAVE_ITER > dist_params.PREFETCH_COUNT:
                self.feed_data(self.t + dist_params.PREFETCH_COUNT)

    def remove_var_at_t(self, t):
        remove_var_ids = []
        remove_ts = []
        for key_t, var_ids in self.reserved_vars_by_t.items():
            if key_t <= t:
                remove_var_ids.extend(var_ids)
                remove_ts.append(key_t)
        for var_id in remove_var_ids:
            varserverapi.remove_var(self.varserver_db, var_id)
        for key_t in remove_ts:
            del self.reserved_vars_by_t[key_t]

    
    def start_training(self):
        # load initial data and start iteration
        logging.info("Start training")
        self.start_iteration()

    def start_iteration(self):
        logging.info("Stroing current weight")
        try:
            self.weight_id = varserverapi.reserve_write_var(self.varserver_db, self.weights_bin)
            self.reserved_vars_by_t[self.t].append(self.weight_id)

            #if self.t % dist_params.MODEL_SAVE_ITER == 0:
            self.save_model()
            logging.info("Start iteration {} weight {}".format(self.t, self.weight_id))
            self.gradient_ids = []
            for client_id in range(dist_params.N_CLIENTS):
                client = self.clients[client_id]
                gradient_id = varserverapi.reserve_var(self.varserver_db)
                self.gradient_ids.append(gradient_id)
                self.reserved_vars_by_t[self.t + dist_params.MODEL_SAVE_ITER].append(gradient_id)
                client.sendMessage(unicode(json.dumps({"command":"calc_gradient", "iterations": dist_params.MODEL_SAVE_ITER,
                "vars":{"weight":self.weight_id, "gradient":gradient_id},
                "lr": dist_params.LR})))
            for pf in range(dist_params.PREFETCH_COUNT + 1):
                self.feed_data(self.t + pf)
        except Exception as ex:
            print(ex)
    
    def save_model(self):
        try:
            logging.info("Saving model at {}".format(self.t))
            with open(self.weight_save_path_tmpl % self.t, "wb") as f:
                f.write(self.weights_bin)
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
