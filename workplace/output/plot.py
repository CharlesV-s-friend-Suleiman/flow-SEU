import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = '../rl_env/data/fixedtime_cav_20240306-1102251709694145.0414932-0_emission.csv'
OUTPUT_PATH = './Figur'
VEH_LIST = [str]


def compute(file_path, type_veh: str, method: str):
    """
    type_veh legal: human or cav ids
    method legal: velocity, headway, 
    """

    raw_df = pd.read_csv(file_path)
    df = raw_df[raw_df["id"] == type_veh]
    res = None
    # draw average velocity
    if method == 'velocity':
        df_velocity = df['speed']
        res = df_velocity.describe()

    if method == 'headway':
        df_headway = df['headway']
        res = df_headway.describe()
    print(res)

    return 0


def plot_v_t(file_path, type_veh: str, method:str):

    raw_df = pd.read_csv(file_path)
    df = raw_df[raw_df["id"].apply(lambda x: x[0:3] == type_veh)]
    grouped_by_ids_df = df.groupby(by=['id'])
    fig = plt.figure(figsize=(10, 8))

    for name, sub_df in grouped_by_ids_df:
        x = None
        y = None
        if method == 'v-t':
            x = sub_df['time']
            y = sub_df['speed']
            plt.axis([0, 500, 0, 30])

        if method == 'x-y':
            x = sub_df['x']
            y = sub_df['y']
        plt.plot(x, y)

    if method == 'v-t':
        xl = 'time of simulation'
        yl = 'velocity of ' + type_veh
        plt.xlabel(xl)
        plt.ylabel(yl)

    fig.savefig(OUTPUT_PATH+'/' + method +'of' + type_veh+'.png')

    return 0


'''compute(FILE_PATH,
        type_veh='cav_0',
        method='velocity')

compute(FILE_PATH,
        type_veh='cav_0',
        method='headway')'''


plot_v_t(FILE_PATH,
         type_veh='cav',method='x-y')

plot_v_t(FILE_PATH,
         type_veh='hum',method='x-y')

plot_v_t(FILE_PATH,
         type_veh='cav',method='v-t')

plot_v_t(FILE_PATH,
         type_veh='hum',method='v-t')