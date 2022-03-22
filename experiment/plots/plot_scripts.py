import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
def plot_figure(file_path, x_axis_col, y_axis_col,x_label,y_label, sub_plt,title):
    data = pd.read_csv(file_path)
    x_data = data[x_axis_col].to_numpy()
    y_data = data[y_axis_col].to_numpy()
    if(y_axis_col=="succs"):
        total_clients = y_data[0]+data["failed"][0]
        y_data = y_data/total_clients
        print(y_label+ " mean",np.mean(y_data))
    # data.plot(kind='line',x=x_axis_col,y=y_axis_col,ax=plt.gca())
    sub_plt.set_xlabel(x_label)
    sub_plt.set_ylabel(y_label)
    sub_plt.set_title(title)
    sub_plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    # sub_plt.set_xticks()
    # sub_plt.set_yticks(np.arange(0,100,10))
    sub_plt.plot(x_data,y_data)

def get_dir_session_files(dir_path):
    files = os.listdir(dir_path)
    inv_file = None
    timing_file = None
    clients_log_file = None
    
    for file in files:
        if file.startswith("invocation_"):
            inv_file = os.path.join(dir_path, file)
        if file.startswith("timing_"):
            timing_file = os.path.join(dir_path, file)
        if file.startswith("clients"):
            clients_log_file = os.path.join(dir_path, file)
        
    return inv_file, timing_file, clients_log_file

def plot_variance(clients_log_path, plot):
    data = pd.read_csv(clients_log_path)
    grouped_data= data.groupby(['client_id']).size()
    print("num clients:", len(grouped_data))
    # print(grouped_data[0])
    plot.set_xlabel("client index")
    plot.set_ylabel("invocations during training")
    plot.set_title("Client Invocations")
    plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot.plot( np.arange(len(grouped_data)),grouped_data)
    return max(grouped_data),min(grouped_data)


def plot_dataset(path,eur_plot,acc_plot,loss_plot, time_plot,var_plot, algorithm_name, dataset_name):
    # use plot list for figures in order
    # use dataset name as legends
    x_labels = [("round_id", "round number"),("round_id", "round number"),("round_id", "round number"),("round_id", "round number")]
    y_labels = [("succs", "EUR"),("global_test_accuracy","accuracy"),("global_test_loss","Loss"),("clients_finished_seconds","training duration")]
    graph_titles = ["Effective Update Ratio", "Test Accuracy","Loss","Round Duration"]
    inv_path,timing_path,clients_log_path = get_dir_session_files(path)
    ## this function plots the variance too
    max_norm, min_norm = plot_variance(clients_log_path,var_plot)
    print(f'{algorithm_name}: variance: {max_norm-min_norm} for max of {max_norm} and min of {min_norm}')
    ## plot eur 
    plot_figure(inv_path,x_labels[0][0],y_labels[0][0],x_labels[0][1],y_labels[0][1],eur_plot,f'{graph_titles[0]}')
    ## plot accuracy
    plot_figure(timing_path,x_labels[1][0],y_labels[1][0],x_labels[1][1],y_labels[1][1],acc_plot,f'{graph_titles[1]}')
    
    ## plot loss time 
    plot_figure(timing_path,x_labels[2][0],y_labels[2][0],x_labels[2][1],y_labels[2][1],loss_plot,f'{graph_titles[2]}')
    
    ## plot time 
    plot_figure(timing_path,x_labels[3][0],y_labels[3][0],x_labels[3][1],y_labels[3][1],time_plot,f'{graph_titles[3]}')
    

            
def plot_dataset_compare_3(path_normals,path_enhanced,path_prox,titles):
    # labels have col name and its name on the grapgh
    x_labels = [("round_id", "round number"),("round_id", "round number")]
    y_labels = [("succs", "EUR"),("global_test_accuracy","accuracy")]
    graph_titles = ["Effective Update Ratio", "Test Accuracy"]
    acc, acc_plts = plt.subplots(1, len(path_normals),figsize=(14,4))
    eur, eur_plts = plt.subplots(1,len(path_normals),figsize=(14,4))
    loss_p, loss_plts = plt.subplots(1, len(path_normals),figsize=(14,4))
    tim, time_plts = plt.subplots(1, len(path_normals),figsize=(14,4))
    var, variance_plts = plt.subplots(1,len(path_normals),figsize=(14,4))
    
    if(len(path_normals)<=1):
        acc_plts = [acc_plts]
        eur_plts = [eur_plts]
        loss_plts = [loss_plts]
        time_plts = [time_plts]
        variance_plts = [variance_plts]
    acc.tight_layout(h_pad=6)
    eur.tight_layout(h_pad=6)
    loss_p.tight_layout(h_pad=6)
    tim.tight_layout(h_pad=6)
    var.tight_layout(h_pad=6)
    for idx, (normal,enhanced,prox) in enumerate(zip(path_normals,path_enhanced,path_prox)):
        plot_dataset(normal,eur_plts[idx],acc_plts[idx],loss_plts[idx],time_plts[idx],variance_plts[idx],"normal","speech")
        plot_dataset(enhanced,eur_plts[idx],acc_plts[idx],loss_plts[idx],time_plts[idx],variance_plts[idx],"enhanced","speech")
        plot_dataset(prox,eur_plts[idx],acc_plts[idx],loss_plts[idx],time_plts[idx],variance_plts[idx],"prox","speech")