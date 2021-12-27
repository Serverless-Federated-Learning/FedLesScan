
from abc import ABC, abstractmethod
import logging
from typing import List
import numpy as np

import random
from requests.sessions import session

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
        # dict of {cluster_idx: (ema,len(cluster))}
        cluster_number_map = {}
        for idx in range(len(labels)):
            client_cluster_idx = labels[idx]
            client_ema = clients[idx].ema
            if client_cluster_idx in cluster_number_map: 
                old_cluster_data = cluster_number_map[client_cluster_idx]
                cluster_number_map[client_cluster_idx] = (old_cluster_data[0]+client_ema,old_cluster_data[1]+1)
            else:
                cluster_number_map[client_cluster_idx] = (client_ema,1)
        
        sorted_clusters = dict(sorted(cluster_number_map, key= lambda x:x[1][1])) 
        
        return sorted_clusters
        # pass
       
        
    def db_fit(self, n_clients_in_round: int, pool: List, round,max_rounds) -> List:
        # Generate sample data
        history_dao = ClientHistoryDao(db = self.config.database_config)
        # all data of the session
        all_data = history_dao.load_all(session_id=self.session)
        # did not run pool
        
        training_data = []
        clients_ids = []
        for client_data in all_data:
            client_training_times = client_data.training_times
            client_ema = client_data.ema
            client_latest_updated = client_data.latest_updated
            rounds_completed = len(client_training_times)
            latest_ema = client_ema
            
            # the ema is not up to date
            if client_latest_updated < rounds_completed-1:
                latest_ema = self.compute_ema(training_times = client_training_times, latest_ema = client_ema,latest_updated = client_latest_updated)
                # load client data and update it
                client_history = history_dao.load(client_data.client_id)
                # compute ema
                client_history.ema = latest_ema
                # update latest index used in ema zero based
                client_history.latest_updated = rounds_completed -1
                history_dao.save(client_history)
            # assume untrained clients have zero training time
            # this way we can run all clients at least once then decisions can be made based on fairness
            
            training_data.append([latest_ema])
            clients_ids.append(client_data.client_id)
        
        # generate this data from db
        X = StandardScaler().fit_transform(training_data)
        # #############################################################################
        # Compute DBSCAN 
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        labels = db.labels_
        
        sorted_clusters = self.sort_clusters(all_data,labels)
        number_of_clusters = len(sorted_clusters)
        #TODO
        runnable_cluster_idx = (round*number_of_clusters)//max_rounds
        
        

        
        
        return random.sample(pool, n_clients_in_round)
        
# db = DBScanClientSelection()     
# db.select_clients([],[])
    
    
    
