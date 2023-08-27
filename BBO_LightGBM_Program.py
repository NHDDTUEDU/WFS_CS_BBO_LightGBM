import tkinter as tk
import numpy as np
from numpy import genfromtxt
import lightgbm as lgb
#-----------------------------------------------------
def ZScoreNorm(X = np.random.random((20, 2))*10):
    # Z score normalization
    MeanX = np.mean(X, axis = 0)
    StdX  = np.std(X, axis = 0)
    nX = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            nX[i,j] = (X[i,j] - MeanX[j])/StdX[j]
    return nX, MeanX, StdX
# ----------------------------------------------------
def ZScoreNormY(X = np.random.random(20)):
    # Z score normalization
    MeanX = np.mean(X)
    StdX  = np.std(X)
    nX = np.zeros(X.shape[0])
    for i in range(X.shape[0]):        
        nX[i] = (X[i] - MeanX)/StdX
    return nX, MeanX, StdX
# ----------------------------------------------------
def ComputeSum():    
    DataLoc = 'WFS_CS.csv'
    dataset	= genfromtxt(DataLoc, delimiter=',')
    Dim = dataset.shape[1]
    X0	= dataset[:,0:Dim-1]
    Y0	= dataset[:,-1]

    Nd = len(Y0)
    ridx = np.random.permutation(Nd)
    X0 = X0[ridx, :]
    Y0 = Y0[ridx]

    X_train, meanX, stdX = ZScoreNorm(X0)
    Y_train, meanY, stdY = ZScoreNormY(Y0)

    Cement_val = float(ent_value_Cement.get())
    SilicaFume_val = float(ent_value_SilicaFume.get()) 
    FlyAsh_val = float(ent_value_FlyAsh.get()) 
    Water_val = float(ent_value_Water.get()) 
    FineAggregate_val = float(ent_value_FineAggregate.get())
    WFS_val = float(ent_value_WFS.get())
    CoarseAggregate_val = float(ent_value_CoarseAggregate.get())
    RCA_val = float(ent_value_RCA.get())
    SP_val = float(ent_value_SP.get())
    CuringTime_val = float(ent_value_CuringTime.get())

    X_queried = np.array([[Cement_val, SilicaFume_val, FlyAsh_val, Water_val, FineAggregate_val, WFS_val, CoarseAggregate_val,\
                           RCA_val, SP_val, CuringTime_val]])
    print('X_queries = ', X_queried)

    X_queried_normalized = ((X_queried - meanX)/stdX) # .reshape((1, len(X_queried)))
    print('X_queried_normalized', X_queried_normalized)
       

    LossFunType_val = sel_loss_func()
    Yp = 0
    if LossFunType_val == 0:
        PredictionModel = lgb.Booster(model_file='Trained_LightGBM_Model.json')
        Yp_normalized = PredictionModel.predict(X_queried_normalized)        
        Yp = Yp_normalized * stdY + meanY
    else:
        PredictionModel = lgb.Booster(model_file='Trained_ASL_LightGBM_Model.json')
        Yp_normalized = PredictionModel.predict(X_queried_normalized)
        Yp = Yp_normalized * stdY + meanY        
   
    ent_value_CS.insert(0, str(Yp[0]))
#----------------------------------------------------------
def sel_loss_func():
    LossFunType_val = LossFunType.get()
    return LossFunType_val
#----------------------------------------------------------
# Set up the window
window = tk.Tk()
window.title("Program")
window.resizable(width=False, height=False)

frm_entry = tk.Frame(master=window)

X_input = [350,	0,	0,	183.24,	712.6,	88.4,	1008,	0,	0.524,	28] # Actual CS = 44.98

# Label title
label_separate = tk.Label(master=frm_entry, text='BBO-LightGBM for Predicting', font='Helvetica 11 bold')
label_separate.grid(row=0, column=0, sticky="w")

label_separate = tk.Label(master=frm_entry, text='CS of Concrete Utilizing WFS', font='Helvetica 11 bold')
label_separate.grid(row=0, column=1, sticky="w")

## Variable Cement
label_name_Cement = tk.Label(master=frm_entry, text="Cement quantity ")
label_name_Cement.grid(row=1, column=0, sticky="w")

ent_value_Cement = tk.Entry(master=frm_entry, width=10)
ent_value_Cement.grid(row=1, column=1, sticky="e")
ent_value_Cement.insert(0, X_input[0])

label_unit_Cement = tk.Label(master=frm_entry, text= 'kg/m\u00B3') 
label_unit_Cement.grid(row=1, column=2, sticky="e")

## Variable SilicaFume
label_name_SilicaFume = tk.Label(master=frm_entry, text="Silica fume quantity ")
label_name_SilicaFume.grid(row=2, column=0, sticky="w")

ent_value_SilicaFume = tk.Entry(master=frm_entry, width=10)
ent_value_SilicaFume.grid(row=2, column=1, sticky="e")
ent_value_SilicaFume.insert(0, X_input[1])

label_unit_SilicaFume = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_SilicaFume.grid(row=2, column=2, sticky="e")

## Variable FlyAsh
label_name_FlyAsh = tk.Label(master=frm_entry, text="Fly ash quantity ")
label_name_FlyAsh.grid(row=3, column=0, sticky="w")

ent_value_FlyAsh = tk.Entry(master=frm_entry, width=10)
ent_value_FlyAsh.grid(row=3, column=1, sticky="e")
ent_value_FlyAsh.insert(0, X_input[2])

label_unit_FlyAsh = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_FlyAsh.grid(row=3, column=2, sticky="e")

## Variable water
label_name_Water = tk.Label(master=frm_entry, text="Water quantity ")
label_name_Water.grid(row=4, column=0, sticky="w")

ent_value_Water = tk.Entry(master=frm_entry, width=10)
ent_value_Water.grid(row=4, column=1, sticky="e")
ent_value_Water.insert(0, X_input[3])

label_unit_Water = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_Water.grid(row=4, column=2, sticky="e")

## Variable FineAggregate
label_name_FineAggregate = tk.Label(master=frm_entry, text="Fine aggregate quantity ")
label_name_FineAggregate.grid(row=5, column=0, sticky="w")

ent_value_FineAggregate = tk.Entry(master=frm_entry, width=10)
ent_value_FineAggregate.grid(row=5, column=1, sticky="e")
ent_value_FineAggregate.insert(0, X_input[4])

label_unit_FineAggregate = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_FineAggregate.grid(row=5, column=2, sticky="e")

## Variable WFS
label_name_WFS = tk.Label(master=frm_entry, text="WFS quantity ")
label_name_WFS.grid(row=6, column=0, sticky="w")

ent_value_WFS = tk.Entry(master=frm_entry, width=10)
ent_value_WFS.grid(row=6, column=1, sticky="e")
ent_value_WFS.insert(0, X_input[5])

label_unit_WFS = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_WFS.grid(row=6, column=2, sticky="e")

## Variable CoarseAggregate
label_name_CoarseAggregate = tk.Label(master=frm_entry, text="Coarse aggregate quantity ")
label_name_CoarseAggregate.grid(row=7, column=0, sticky="w")

ent_value_CoarseAggregate = tk.Entry(master=frm_entry, width=10)
ent_value_CoarseAggregate.grid(row=7, column=1, sticky="e")
ent_value_CoarseAggregate.insert(0, X_input[6])

label_unit_CoarseAggregate = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_CoarseAggregate.grid(row=7, column=2, sticky="e")

## Variable RCA
label_name_RCA = tk.Label(master=frm_entry, text="Recycled coarse aggregate quantity ")
label_name_RCA.grid(row=8, column=0, sticky="w")

ent_value_RCA = tk.Entry(master=frm_entry, width=10)
ent_value_RCA.grid(row=8, column=1, sticky="e")
ent_value_RCA.insert(0, X_input[7])

label_unit_RCA = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_RCA.grid(row=8, column=2, sticky="e")

# Variable SP
label_name_SP = tk.Label(master=frm_entry, text="Superplasticizer quantity ")
label_name_SP.grid(row=9, column=0, sticky="w")

ent_value_SP = tk.Entry(master=frm_entry, width=10)
ent_value_SP.grid(row=9, column=1, sticky="e")
ent_value_SP.insert(0, X_input[8])

label_unit_SP = tk.Label(master=frm_entry, text='kg/m\u00B3')
label_unit_SP.grid(row=9, column=2, sticky="e")

# Variable CuringTime
label_name_CuringTime = tk.Label(master=frm_entry, text="Time ")
label_name_CuringTime.grid(row=10, column=0, sticky="w")

ent_value_CuringTime = tk.Entry(master=frm_entry, width=10)
ent_value_CuringTime.grid(row=10, column=1, sticky="e")
ent_value_CuringTime.insert(0, X_input[9])

label_unit_CuringTime = tk.Label(master=frm_entry, text='day')
label_unit_CuringTime.grid(row=10, column=2, sticky="w")

#
label_SelectLossFunctionType = tk.Label(master=frm_entry, text='Loss function', font='Helvetica 10 bold')
label_SelectLossFunctionType.grid(row=11, column=0, sticky="w")

# Variable CS
label_name_CS = tk.Label(master=frm_entry, text="Compressive strength", font='Helvetica 10 bold')
label_name_CS.grid(row=15, column=0, sticky="w")

ent_value_CS = tk.Entry(master=frm_entry, width=10)
ent_value_CS.grid(row=15, column=1, sticky="e")
ent_value_CS.insert(0, '')

label_unit_CS = tk.Label(master=frm_entry, text='MPa')
label_unit_CS.grid(row=15, column=2, sticky="w")

#----------------------------------
label_separate = tk.Label(master=frm_entry, text='')
label_separate.grid(row=16, column=0, sticky="n")
##
##label_separate = tk.Label(master=frm_entry, text='')
##label_separate.grid(row=12, column=0, sticky="n")
##
##label_separate = tk.Label(master=frm_entry, text='')
##label_separate.grid(row=14, column=0, sticky="n")
#----------------------------------
btn_ComputeSum = tk.Button(
    master=frm_entry,   
    text = 'Compute',
    command=ComputeSum
)
btn_ComputeSum.grid(row=12, column=1, sticky="e")
#----------------------------------
LossFunType = tk.IntVar()
R1 = tk.Radiobutton(master=frm_entry, text="SEL", variable=LossFunType, value=0,
                  command=sel_loss_func)
R1.grid(row=12, column=0, sticky="w")

R2 = tk.Radiobutton(master=frm_entry, text="ASEL", variable=LossFunType, value=1,
                  command=sel_loss_func)
R2.grid(row=13, column=0, sticky="w")

#----------------------------------

# Set up the layout using the .grid() geometry manager
frm_entry.grid(row=0, column=0, padx=10)

# Run the application
window.mainloop()
