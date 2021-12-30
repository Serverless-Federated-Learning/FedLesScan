
from abc import ABC, abstractmethod
import logging
from typing import List
import numpy as np

import random
from requests.sessions import session
from sklearn import cluster

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from fedless.benchmark.models import ExperimentConfig

from fedless.persistence.client_daos import ClientHistoryDao

logger = logging.getLogger(__name__)


class IntelligentClientSelection(ABC):

    def __init__(self, execution_func,function_params = {}):
        """
        execution_func: function to execute the selection procedure
        function_params: parameters for the function to be called
        """
        self.excution_func = execution_func
        self.function_params = function_params 
        
    def select_clients(self, n_clients_in_round: int, clients_pool: List, round, max_rounds) -> List:
        """
        :clients: number of clients
        :pool: List[ClientConfig] contains each client necessary info for selection
        :func: function to order the clients
        :return: list of clients selected from the pool based on selection criteria 
        """
        return self.excution_func(n_clients_in_round, clients_pool,round, max_rounds, **self.function_params)

class RandomClientSelection(IntelligentClientSelection):
    def __init__(self):
        super().__init__(self.sample_clients)
        
    def sample_clients(self, clients: int, pool: List) -> List:
        return random.sample(pool, clients)

class DBScanClientSelection(IntelligentClientSelection):
    
    def __init__(self, config: ExperimentConfig, session):
        super().__init__(self.db_fit)
        self.config = config
        self.session = session
    
    
    def compute_ema(self, training_times:list, latest_ema:float,latest_updated:int, smoothingFactor: float = 0.5):
        """
        Parameters
        ----------
        training_times : list
            The name of the animal
        latest_ema : float
            The last ema computation
        latest_update_idx : int, optional
            the last idx for ema computed before
        """
        updated_ema = latest_ema
        for i in range(latest_updated+1,len(training_times)):
            updated_ema = updated_ema*smoothingFactor+ training_times[i]
        return updated_ema
    
    def sort_clusters(self, clients:list,labels:list):
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # dict of {cluster_idx: (ema,[client_list])}
        cluster_number_map = {}
        for idx in range(len(labels)):
            client_cluster_idx = labels[idx]
            client_ema = clients[idx].ema
            if client_cluster_idx in cluster_number_map: 
                old_cluster_data = cluster_number_map[client_cluster_idx]
                # append new client
                old_cluster_data[1].append(clients[idx])
                cluster_number_map[client_cluster_idx] = (old_cluster_data[0] + client_ema, old_cluster_data[1])
            else:
                cluster_number_map[client_cluster_idx] = (client_ema,[clients[idx]])
        # sort clusters based on avg ema per cluster 
        return dict(sorted(cluster_number_map.items(), key= lambda x:x[1][0]/len(x[1][1]))) 
        # pass
    
    def filter_rookies(self, clients:list):
        rookies = []
        rest_clients = []
        for client in clients:
            if len(client.training_times)==0 and len(client.missed_rounds) == 0:
                rookies.append(client)
            else:
                rest_clients.append(client)
            #   filter(lambda item: len(item.training_times)==0 and len(item.missed_rounds) == 0,clients)  
        return rookies,rest_clients
        
                
    def db_fit(self, n_clients_in_round: int, pool: List, round,max_rounds) -> list:
        # Generate sample data
        history_dao = ClientHistoryDao(db = self.config.database)
        # all data of the session
        all_data = list(history_dao.load_all(session_id=self.session))
        
        # try and run rookies first
        rookie_clients,rest_clients = self.filter_rookies(all_data)    
        if len(rookie_clients)>= n_clients_in_round:
            return random.sample(rookie_clients,n_clients_in_round)
        
        n_clients_from_clustering = n_clients_in_round-len(rookie_clients)
        
        training_data = []
        for client_data in rest_clients:
            client_training_times = client_data.training_times
            client_ema = client_data.ema
            client_latest_updated = client_data.latest_updated
            rounds_completed = len(client_training_times)
            latest_ema = client_ema
            
            # the ema is not up to date
            if client_latest_updated < rounds_completed-1:
                latest_ema = self.compute_ema(training_times = client_training_times, latest_ema = client_ema,latest_updated = client_latest_updated)
                # load client data and update it
                # client_history = history_dao.load(client_data.client_id)
                # compute ema
                # client_history.ema = latest_ema
                # update latest index used in ema zero based
                # client_history.latest_updated = rounds_completed -1
                # history_dao.save(client_history)
                client_data.ema = latest_ema
                client_data.latest_updated = rounds_completed - 1
                history_dao.save(client_data)
            training_data.append([latest_ema])
        
        # generate this data from db
        X = StandardScaler().fit_transform(training_data)
        # #############################################################################
        # Compute DBSCAN 
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        labels = db.labels_
        sorted_clusters = self.sort_clusters(rest_clients,labels)
        cluster_idx_list = np.arange(start = 0, stop = len(sorted_clusters))
        perc = (round/max_rounds)*100
        start_cluster_idx = np.percentile(cluster_idx_list,perc,interpolation='nearest')
        
        return rookie_clients + self.sample_starting_from(sorted_clusters,start_cluster_idx, n_clients_from_clustering)

        
        
    
    def sample_starting_from(self,sorted_clusters,start_cluster_idx:int, n_clients_from_clustering:int)->list:
        # return clients which run the least
        cluster_list = list(sorted_clusters.items())
        returned_samples = []
        while n_clients_from_clustering>0:
            cluster = cluster_list[start_cluster_idx]
            cluster_clients = cluster[1][1]
            cluster_size = len(cluster_clients)
            if cluster_size >= n_clients_from_clustering:
                cluster_clients_sorted = sorted(cluster_clients,key = lambda client: len(client.training_times))
                return cluster_clients_sorted[:n_clients_from_clustering]
            else:
                n_clients_from_clustering-= cluster_size
                returned_samples+= cluster_clients
            # if clusters are done go back and fetch from the faster clients    
            start_cluster_idx = (start_cluster_idx+1) % len(cluster_list)
            
        return returned_samples            
        # return random.sample(pool, n_clients_in_round)
        
        pass
        
# db = DBScanClientSelection()     
# db.select_clients([],[])
    
    
    
