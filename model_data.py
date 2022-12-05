import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t



def historgram_figure(model,norm=False):
    labels = np.load(f"{model}_labels.npy")
    final_output = np.load(f"{model}_output.npy")
    time_step = 50
    
    final_output = final_output.reshape(-1,time_step,16)
    labels = labels.reshape(-1,time_step,16)

    bins_vals = []
    avg = np.average(np.absolute(labels))
    for pred, val in zip(final_output, labels):
        x = np.sqrt(np.square(np.subtract(pred, val)).mean())
        if norm: x /= avg
        bins_vals.append(x)

    bins_vals = np.transpose(np.array(bins_vals))
    print(bins_vals.shape)
    print(bins_vals)
    #fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    #axs.hist(bins_vals, bins=10)
    #plt.xticks(np.arange(0,30,2))
    
    x = np.arange(0, 5,0.2) if norm else np.arange(0, 2000,100)
    
    plt.figure(figsize=(13,5))
    _, _, bars = plt.hist(bins_vals, x)
    plt.xticks(x)
    title = f"{model} Normalized RMSE for each Time Step Histogram" if norm else f"{model} RMSE for each Time Step Histogram"
    plt.title(title)
    plt.bar_label(bars)
    title = f'{model}_figs/RNN_histogram_norm.png' if norm else f'{model}_figs/RNN_histogram.png'
    plt.savefig(title, dpi=200)
    plt.close()


def actual_target_ouput(model, norm=False):
    labels = np.load(f"{model}_labels.npy")
    final_output = np.load(f"{model}_output.npy")
    time_step = 50
    channel = 8
    start = 2000
    end = 2200
    if norm:
            final_output /= np.average(np.absolute(labels))
            labels /= np.average(np.absolute(labels))
    for channel in range(0,16):
        final_output_seq = final_output[start:end,channel:channel+1]
        labels_seq = labels[start:end,channel:channel+1]
        #inputs = inputs[1000:1100,0]
        
        #plt.figure(figsize=(,5))

        x = np.arange(0, end-start,50)
        plt.xticks(x)
        plt.plot(final_output_seq,label="Output")
        plt.plot(labels_seq, label="Label")
        #plt.plot(inputs, label="Inputs")
        plt.xlabel("Time")
        plt.ylabel("mVolt")
        plt.title(f"{model} Target to Actual Graph from Sample {start}-{end} for Channel {channel}")
        plt.legend()
        title = f'{model}_figs/viz/{model}_predToAcutal_channel{channel}_norm.png' if norm else f'{model}_figs/viz/{model}_predToAcutal_channel{channel}.png'
        plt.savefig(title)
        plt.close()

def z_score(model):
    labels = np.load(f"{model}_labels.npy")
    final_output = np.load(f"{model}_output.npy")

    avg_output = np.average(final_output)
    avg_labels = np.average(labels) 

    std_output = np.std(final_output)
    std_labels = np.std(labels)
    

    z_output = (final_output - avg_output)/std_output
    z_labels = (labels - avg_labels) / std_labels

    plt.plot(z_output[0:100,0])
    plt.plot(z_labels[0:100,0])  
    plt.show()


def CI(model):
    labels = np.load(f"{model}_labels.npy")
    final_output = np.load(f"{model}_output.npy") 
    print(labels[0][0])
    def ci_sub(x, confidence=0.95):
        m = x.mean() 
        s = x.std() 
        dof = len(x)-1 
        t_crit = np.abs(t.ppf((1-confidence)/2,dof))
        return (m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))) 
    
    print(f"{model} Labels CI: {ci_sub(labels)}")
    print(f"{model} Output CI: {ci_sub(final_output)}")




def loss_fig(model):

    train_loss = np.load(f"{model}_train_loss.npy")
    val_loss = np.load(f"{model}_train_loss.npy")
    epochs = np.load(f"{model}_epochs_axis.npy")
    plt.plot(epochs, train_loss,label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"{model}_figs/Loss", dpi=200)
    plt.close()

#actual_target_ouput()
def stats(model):
    labels = np.load(f"{model}_labels.npy")
    final_output = np.load(f"{model}_output.npy")
    time_step = 50
    print(labels.reshape(-1,1).shape)
    RMSE = np.sqrt(np.square(np.subtract(labels.reshape(1,-1), final_output.reshape(1,-1))).mean())
    Norm_RMSE = RMSE/np.average(np.absolute(labels.reshape(1,-1)))
    print(np.average(np.absolute(labels.reshape(1,-1))))
    print(f"{model} RMSE:",RMSE )
    print(f"{model} Normalized RMSE:",Norm_RMSE)
    print(f"{model} Output Average {np.average(final_output)}")
    print(f"{model} Output Variance {np.var(final_output)}")
    print(f"{model} Label Average {np.average(labels)}")  
    print(f"{model} label Variance {np.var(labels)}")

model = "LSTM"

#z_score(model)

loss_fig(model)
historgram_figure(model,False)
historgram_figure(model,True)
actual_target_ouput(model,True)
actual_target_ouput(model,False)
stats(model)
CI(model)
