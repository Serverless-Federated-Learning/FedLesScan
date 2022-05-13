import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import math
from cost import closestNumber,calculate_costs_gcf
import json
DPI = 300
mpl.rcParams['figure.dpi'] = DPI
axis_font = 14
def plot_figure(file_path, x_axis_col, y_axis_col,x_label,y_label, plt,title,plot_type, line_style="solid",strategy_name = None,color="blue"):
    # line type is a string refering to line 
    data = pd.read_csv(file_path)
    x_data = data[x_axis_col].to_numpy()
    y_data = data[y_axis_col].to_numpy()
    if(y_axis_col=="succs"):
        total_clients = y_data[0]+data["failed"][0]
        straggler_rounds = np.count_nonzero(y_data<total_clients)
        y_data = y_data/total_clients
        print(f'{y_label} mean: {np.mean(y_data)}, straggler rounds: {straggler_rounds}')
        
    fig,sub_plt = plt
    sub_plt.set_xlabel(x_label,fontsize = axis_font)
    sub_plt.set_ylabel(y_label,fontsize = axis_font)
    # sub_plt.set_title(title)
    sub_plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    # sub_plt.set_xticks()
    # sub_plt.set_yticks(np.arange(0,100,10))
    
    l = None
    if plot_type == "line":
      l,=  sub_plt.plot(x_data,y_data,linestyle=line_style,label=strategy_name,linewidth=3,color=color)
    elif plot_type == "violin":
      l,=  sub_plt.violinplot(x_data,y_data,label=strategy_name)
    elif plot_type == "bar":
      l,=  sub_plt.bar(x_data,y_data,label=strategy_name)
    

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

def get_min_max_cost(client_logs_path):
    client_table = pd.read_csv(client_logs_path)
    timings_max= np.array([])
    timings_min = np.array([])
    cost_min = 0
    cost_max = 0
    max_time = 40
    for idx,client in client_table.iterrows():
        client_func = json.loads(client['function'])
        url = client_func['params']['url']
        id = url.split("-")[2]
        sim_delay = client_func['invocation_delay']
        exec_time = client["seconds"]    
            
        if sim_delay ==-1:
           timings_max = np.append(timings_max,max_time)
        else:
            timings_max = np.append(timings_max,exec_time)
        timings_min = np.append(timings_min,exec_time)
        

    cost_list = calculate_costs_gcf(timings_min)
    cost_min = np.sum(cost_list)
    cost_max = np.sum(calculate_costs_gcf(timings_max))
    print("costmin: ", cost_min, " costmax: ", cost_max)
    
    

def plot_variance(grouped_data, plt, title):
    # data = pd.read_csv(clients_log_path)
    # grouped_data= data.groupby(['client_id']).size()
    # print("num clients:", len(grouped_data))
    # print(grouped_data[0])
    fig,plot = plt
    # plot.set_xlabel("client index")
    plot.set_ylabel("Invocations")
    plot.set_title(title)
    # plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot.set_xticks([1,2,3])
    labels=["FedAvg","FedlesScan","Fedprox"]
    plot.set_xticklabels(labels)
    # plot.xaxis.set_tick_params(direction='out')
    # plot.xaxis.set_ticks_position('bottom')
    # plot.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    plot.set_xlim(0.25, len(labels) + 0.75)
    
    plot.violinplot(grouped_data,[1,2,3])
    

def get_client_inv_group(path):
    _,_,client_log_path = get_dir_session_files(path)
    data = pd.read_csv(client_log_path)
    grouped_data= data.groupby(['client_id']).size()
    print("num clients:", len(grouped_data))

    return grouped_data

def plot_dataset(path,eur_plot,acc_plot,loss_plot, time_plot,var_plot, strategy_name,line_style,stragglers,color):
    # use plot list for figures in order
    # use dataset name as legends
    x_labels = [("round_id", "Round Number"),("round_id", "Round Number"),("round_id", "Round Number"),("round_id", "Round Number")]
    y_labels = [("succs", "EUR"),("global_test_accuracy","Accuracy"),("global_test_loss","Loss"),("clients_finished_seconds","Training Duration")]
    graph_titles = ["Effective Update Ratio", "Test Accuracy","Loss","Round Duration"]
    inv_path,timing_path,clients_log_path = get_dir_session_files(path)
    ## this function plots the variance too
    # max_norm, min_norm = plot_variance(clients_log_path,var_plot)
    # print(f'{algorithm_name}: variance: {max_norm-min_norm} for max of {max_norm} and min of {min_norm}')
    ## plot eur 
    get_min_max_cost(clients_log_path)
    plot_figure(inv_path,x_labels[0][0],y_labels[0][0],x_labels[0][1],y_labels[0][1],eur_plot,stragglers,plot_type="line",line_style=line_style,strategy_name = strategy_name,color=color)
    ## plot accuracy
    plot_figure(timing_path,x_labels[1][0],y_labels[1][0],x_labels[1][1],y_labels[1][1],acc_plot,stragglers,plot_type="line",line_style=line_style,strategy_name = strategy_name,color=color)
    
    ## plot loss time 
    plot_figure(timing_path,x_labels[2][0],y_labels[2][0],x_labels[2][1],y_labels[2][1],loss_plot,stragglers,plot_type="line",line_style=line_style,strategy_name = strategy_name,color=color)
    
    ## plot time 
    plot_figure(timing_path,x_labels[3][0],y_labels[3][0],x_labels[3][1],y_labels[3][1],time_plot,stragglers,plot_type="line",line_style=line_style,strategy_name = strategy_name,color=color)
    print("------------------------------------------------------------------")
    
def get_plot():
    
    cols,rows = (3,2)  #3*2
    # grid = plt.GridSpec(rows, cols, wspace = .25, hspace = .25)
    # plot,subs = plt.subplots(rows, cols,figsize=(12,12),squeeze=False)
    plot = plt.figure(figsize=(12,8))
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2,fig=plot)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2,fig=plot)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2,fig=plot)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2,fig=plot)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2,fig=plot)
    plot.tight_layout(h_pad=4,w_pad=4)
    
    # subs[1][2].set_visible(False)

    # subs[1][0].set_position([0.24,0.125,0.343,0.343])
    # subs[1][1].set_position([0.55,0.125,0.343,0.343])
    return plot,[ax1,ax2,ax3,ax4,ax5]
    
            
def plot_dataset_compare_3(path_normals,path_enhanced,path_prox,titles,dataset_title):
    # labels have col name and its name on the grapgh
    x_labels = [("round_id", "round number"),("round_id", "round number")]
    y_labels = [("succs", "EUR"),("global_test_accuracy","accuracy")]
    graph_titles = ["Effective Update Ratio", "Test Accuracy"]
    cols,rows = (math.ceil(len(path_normals)/2),2)  #3*2
    stragglers_p = ["Normal","10% Stragglers","30% Stragglers","50% Stragglers","70% Stragglers"]
    # cols,rows = len(path_normals),1 
    acc, acc_plts = get_plot()
    eur, eur_plts = get_plot()
    loss_p, loss_plts = get_plot()
    tim, time_plts = get_plot()
    var, variance_plts = get_plot()
    
        
    if(len(path_normals)<=1):
        acc_plts = [acc_plts]
        eur_plts = [eur_plts]
        loss_plts = [loss_plts]
        time_plts = [time_plts]
        variance_plts = [variance_plts]
    # variance plots
    for idx, (normal,enhanced,prox) in enumerate(zip(path_normals,path_enhanced,path_prox)):
        
        plot_dataset(normal,(eur,eur_plts[idx]),(acc,acc_plts[idx]),(loss_p,loss_plts[idx]),(tim,time_plts[idx]),(var,variance_plts[idx]),"FedAvg" if idx==0 else None, line_style="solid",stragglers=stragglers_p[idx],color="black")
        plot_dataset(enhanced,(eur,eur_plts[idx]),(acc,acc_plts[idx]),(loss_p,loss_plts[idx]),(tim,time_plts[idx]),(var,variance_plts[idx]),"FedlesScan" if idx==0 else None,line_style="dashed",stragglers=stragglers_p[idx],color="blue")
        plot_dataset(prox,(eur,eur_plts[idx]),(acc,acc_plts[idx]),(loss_p,loss_plts[idx]),(tim,time_plts[idx]),(var,variance_plts[idx]),"FedProx" if idx==0 else None,line_style="dotted",stragglers=stragglers_p[idx],color="brown")
        
        var_normal,var_e,var_prox = get_client_inv_group(normal),get_client_inv_group(enhanced),get_client_inv_group(prox)
        
        plot_variance([var_normal,var_e,var_prox],(var,variance_plts[idx]),title=stragglers_p[idx])
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")


        
        
        # variance plots here

            
    leg = acc.legend(loc='lower right')
    leg = eur.legend(loc='lower right')
    leg = loss_p.legend(loc='lower right')
    leg = tim.legend(loc='lower right')
    leg = var.legend(loc='lower right')
    path = f"./figs/{dataset_title}"
    
    
    # acc.savefig(f'{path}/acc_{dataset_title}.pdf', bbox_inches='tight',dpi = DPI)
    # eur.savefig(f'{path}/eur_{dataset_title}.pdf', bbox_inches='tight',dpi = DPI)
    # loss_p.savefig(f'{path}/loss_{dataset_title}.pdf', bbox_inches='tight',dpi = DPI)
    # tim.savefig(f'{path}/tim_{dataset_title}.pdf', bbox_inches='tight',dpi = DPI)
    # var.savefig(f'{path}/var_{dataset_title}.pdf', bbox_inches='tight',dpi = DPI)

