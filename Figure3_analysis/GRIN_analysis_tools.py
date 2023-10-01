import numpy as np
from numpy import trapz
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from  matplotlib.colors import LinearSegmentedColormap
from path import Path

from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale, scale
from sklearn.metrics import auc
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
from scipy.integrate import simps
from scipy.stats import ks_2samp
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')
from warnings import warn
from collections import OrderedDict
import itertools



def split_recordings(path):
    '''
    Separates multiple recordings that have been concatenated into the same .csv file during longitudinal
    registration or time series creation by IDPS. It requires the presence of a .csv file in the same directory
    which details the names or conditions under which each recording in the concatenated file was made.
    Rows in the log file are recordings, while columns are different experimental conditions. Column titles
    can be any variable that specifies your recordings and there can be as many columns as needed, but
    there should be at least one. The number of rows should match the number of recordings in the file, otherwise
    some videos will be merged.

    Example:

    state                      test

    baseline                   resident_intruder
    baseline                   tube_test
    hungry                     resident_intruder
    hungry                     tube_test

    The name of the log file should be the same as the recording file with the addition of '_log.csv' at the end.
    '''

    path = Path(path)
    directory = path.parent
    filename = path.stem

    # Load from .csv
    df = pd.read_csv(path, index_col = 0)

    # Find and load log .csv
    log_path = directory + '/' + filename + '_log.csv'
    log = pd.read_csv(log_path)

    conditions = log.columns

    # Find boundaries between concatenated videos
    timestamps = df.iloc[1:, 0].astype(float)

    frame_time_diff = timestamps.diff().mode()
    boundaries = timestamps.diff().round(decimals=3) != frame_time_diff[0].round(decimals=3)
    boundaries = np.where(boundaries)[0][1:]

    boundaries += 1

    # Separate recordings and save as .csv
    intervals = np.append(np.insert(boundaries.repeat(2), 0, [1]), df.shape[0]).reshape(-1, 2)

    for n, interval in enumerate(intervals):

        save_path = directory + '/' + filename

        for condition in conditions:
            save_path += '_' + log.loc[n, condition]

        recording = df.iloc[interval[0]:interval[1], :]
        recording.loc[-1] = df.loc[0]
        recording.index = recording.index + 1
        recording.sort_index(inplace=True)
        recording.to_csv(save_path + '.csv', index=False)


def load_inscopix(path, normalisation= None):
    '''
    Loads neuron activity traces from csv file and normalises each cell to itself to interval [0,1] for illustration purposes

    Args:
        path: str; path to Inscopix result .csv file
        normalisation: str; type of normalisation to run on the data. ['absent', 'z-score', 'fixed interval']

    Returns:
        df: np.array; an NxM matrix, where N are cells and M are timepoints in the recording
    '''

    df = pd.read_csv(path)

    df = df.iloc[:, 1:]
    df = df.loc[1:, df.iloc[0] == 'accepted']

    if df.empty:
        df = pd.read_csv(path)
        df = df.iloc[:, 1:]
        df = df.loc[1:, df.iloc[0] == ' accepted']

    df = np.array(df).astype(float)
    #df = np.nan_to_num(df,nan=0.0)

    if normalisation == 'z-score':
        df = stats.zscore(df, axis=0)

    elif normalisation == 'fixed interval':
        for n, cell in enumerate(df.T):
            f = interp1d([cell.min(), cell.max()], [0, 1])
            cell = f(cell)

            df[:, n] = cell


    return df.T


def interpld(path, inscopix_len, header=15):
    '''

    Imports behaviour data from BORIS .csv file and constructs a binary ON/OFF array for each behaviour.

    Args
        path: string; path to the behaviour data .csv file.
        inscopix_len: int; the number of frames in the corresponding Inscopix recording (processed not original).
        header: int; the row number of the behaviour .csv file which contains the column titles. Any rows above this index will be dropped.

    Returns
        num_episodes: Number of behaviour episodes detected
        bdf_p: A pd.DataFrame with binary behaviour arrays where each column is a different behaviour
        events: A list of behaviour event arrays for creating ethograms
    '''

    # Check if argument type is correct
    if type(inscopix_len) is not int:
        raise TypeError("inscopix_len has to be an integer")

    # Import behaviour data
    bdf = pd.read_csv(path, header=header)

    # Interpolate from behaviour times to Inscopix frames
    f = interp1d([0, bdf['Total length'][0]], [0, inscopix_len])
    bdf['Time'] = f(bdf['Time'])
    return bdf

def preprocess_behaviour(path, inscopix_len):

    # Import interpld behaviour data
    bdf = pd.read_csv(path)

    # Extract behaviour names
    behavs = bdf['Behavior'].unique()

    # Initialise empty results variable
    num_episodes = pd.DataFrame()
    bdf_p = pd.DataFrame()
    events = []

    # Construct a binary ON/OFF array for each behaviour
    for n, behav in enumerate(behavs):
        behav = bdf.loc[bdf['Behavior'] == behav, :]

        # Extract behaviour start and stop indices expressed in inscopix frames
        starts = behav['Time'][behav['Status'] == 'START']
        starts = starts.astype(int)

        stops = behav['Time'][behav['Status'] == 'STOP']
        stops = stops.astype(int)

        binary = np.zeros(inscopix_len)

        # Make binary array and store it in the result variable
        if starts.empty:
            point = behav['Time'][behav['Status'] == 'POINT']
            binary[point.astype(int)] = 1

        else:
            for episode in zip(starts, stops):
                start = episode[0]
                stop = episode[1]

                binary[start:stop] = 1

        num_episodes.loc[0, behavs[n]] = len(starts)
        bdf_p[behavs[n]] = binary
        bdf_p = bdf_p.astype(int)

        events.append(binary.nonzero()[0])

    return num_episodes, bdf_p, events

def skip_behav_overlap(bdf, window, fr):

    '''

    identify overlapped behaviour episodes within the identified window and delete the latter of the overlapped
    episode pairs

    Args:

        bdf: a binary pandas dataframe with 0 specifiying absence of a behaviour and 1 specifies presencr of
             a behaviour
       window: around each behavioural event in seconds that is used for analysis
              e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                signal period after the behavioural event


    Returns:

        bdf: a binary pandas dataframe with overlapped behaviour deleted

    '''

    indices = pd.DataFrame(columns = ['start', 'stop', 'original_stop', 'behav'])
    index = {}
    behavs = list(bdf.columns)
    remove_behav = ['nest time', 'male intruder contact', 'female intruder contact']

    behavs = [x for x in behavs if (x not in remove_behav)]

    # extract start and stop indices for all behaviour events
    for behav in behavs:

        start = np.where((bdf[behav].diff() > 0))[0]
        stop = np.where(bdf[behav].diff() < 0)[0]
        index['start'] = [start]
    # obtain stop indices for all behaviour episode other than the ones that associate with the start index
        index['stop'] = [stop[1:]]
        index['original_stop'] = [stop]
        index = pd.DataFrame(index)

    # save the original start and stop indices pair in numpy array indices
        indices = pd.concat([indices, index])

    # obtain start and stop indices pair for every behaviour combination
    indices_ep = indices.explode('start')
    indices_ep = indices_ep.explode('stop')

    # obtain the time difference between the start and stop indices
    for i in range(len(indices_ep.start.unique())):
        start = indices_ep.start.unique()[i]

        for j in range(len(indices_ep.stop.unique())):
            stop = indices_ep.stop.unique()[j]

            diff = start - stop

     # erase behaviour only if the interval is shorter than window *fr
            if diff < window*fr and diff >0:
                deleted = indices_ep[indices_ep.start == start]

                deleted_start = int(deleted['start'].unique()[0])

                behaviour = deleted['behav'].unique()[0]

    # obtain stop index of the behaviour needed to be erased from the original indices array
                deleted_index = indices[indices.behav == behaviour].start[0]

                deleted_index = np.where(deleted_index == deleted_start)[0]

                deleted_stop = indices[indices.behav == behaviour].original_stop[0][deleted_index]

                bdf.loc[int(deleted_start):int(deleted_stop),behaviour] = 0


    return bdf

def extract_behav_episodes(df, bdf, behav, window, fr=20, centered='start'):
    '''
    Extracts segments of the calcium recording corresponding to a user specified window around the start or end of a behavioural episode.

    Args:
        df: np.array; The calcium recording - an MxP matrix, where M are cells and and P are timepoints in the recording
        bdf: pd.DataFrame; a pandas dataframe containing binary behaviour arrays in columns. Column names must be behaviour names.
        behav: str; specifies which behaviour to extract
        window: float; the recording duration before and after behavioural events in seconds that you would like to extract. The final extracted episode will be twice the window duration.
        fr: int; framerate of the recording
        centered: str; 'start' or 'end' if you wish to extract window around the beginning or end of the behaviour episode

    Returns:
        episodes: np.array; an NxMxP matrix, where N is the number of instances of the specified behaviour, M is the number of neurons in df (the recording) and P is the window in frames

    '''

    # Convert window from seconds to frames
    window *= fr
    window = int(window)

    # Identify behavioural events
    if centered == 'start':
        events = np.where(bdf[behav].diff() > 0)[0]
    elif centered == 'end':
        events = np.where(bdf[behav].diff() < 0)[0]
    else:
        raise ValueError("centered argument must be either 'start' or 'end'")

    # Identify behavioural events whose window will reach over the limits of the calcium dataframe
    events = np.delete(events, (events + 50) >= df.shape[1])
    events = np.delete(events, (events - 50) < 0)

    # Prepare result variable
    episodes = np.zeros((len(events), int(df.shape[0]), window * 2))

    # For each behavioural episode extract corresponding calcium activity
    for n, event in enumerate(events):
        episode = np.arange(event - window, event + window)

        if episode[-1] > df.shape[1]:
            continue
        else:
            episode = df[:, episode]
            episodes[n] = episode

    return episodes


def behaviour_episode_zscore(data, baseline_interval):
    '''
    Z scores an array of behaviour episodes using a user-defined baseline period for mean and standard deviation calculation

    Args
        data: np.array; PxNxM matrix, where P are episodes, N are cells and M are timepoints in the behavioural episode.
        baseline_interval: list; [start, end] specifies the start and end indices (in frames) of the baseline interval

    Returns
        zscore: np.array, zscored version of data with the same shape
    '''

    bstart, bend = baseline_interval

    baseline = data[:, :, bstart:bend]
    baseline_mean = baseline.mean(axis=2)
    baseline_std = np.std(baseline, axis=2)

    zscores = np.moveaxis(data, [2], [0]) - baseline_mean
    zscores = zscores / baseline_std
    zscores = np.moveaxis(zscores, [0], [2])

    return zscores


def behaviour_subset(binary, n):
    '''Randomly selects a subset of n behavioural episodes from binary behaviour array

    Args:
        binary: np.array; a binary behaviour array, where presence of behaviour is 1 and absence is 0
        n: int, number of behavioural episodes to select from binary

    Returns:
        subset: np.array; an array of length == len(binary) but containing only the selected subset of behavioural episodes

    '''

    all_starts = np.where(binary.diff() > 0)[0]
    all_stops = np.where(binary.diff() < 0)[0]

    if len(all_starts) < n:
        raise ValueError('The behaviour subset must be smaller than the total number of behavioural episodes.')

    selected = np.random.choice(range(len(all_starts)), n, replace=False)

    starts = all_starts[selected]
    stops = all_stops[selected]

    subset = np.zeros(len(binary))
    for episode in zip(starts, stops):
        start = episode[0]
        stop = episode[1]

        subset[start:stop] = 1

    return subset


def make_design_matrix(data, history=25):
    """
    Create time-lagged design matrix from stimulus intensity vector.

    Args:
        data (1D array): Stimulus intensity at each time point.
        history (number): Number of time lags to use.

    Returns
        X (2D array): GLM design matrix with shape T, d
    """

    # Create version of stimulus vector with zeros before onset
    padded_data = np.concatenate([np.zeros(history - 1), data])

    # Construct a matrix where each row has the d frames of
    # the stimulus proceeding and including timepoint t
    T = len(data)  # Total number of timepoints (number of stimulus frames)
    X = np.zeros((T, history))
    for t in range(T):
        X[t] = padded_data[t:t + history]

    return X


def estimate_global_tuning(df, binary, tuned_to, history=1, cv=8, scale_auc=True, remove_nontarget=False,
                           subtract_shuffled_tuning=True):
    '''

    Estimates the tuning of each cell in dataframe df to specified binary variable by fitting a logistic GLM. Time lags of each
    data point can optionally be used as additional regressors. Tuning is expressed as a 6-fold cross-validated ROC AUC.

    Args
        df: A 2D array of shape NxM, where N are cells and M are samples

        binary: pd.DataFrame; columns are binary ethogram arrays and have string names

        tuned_to: str; name of the behaviour contained in binary for which tuning will be estimated

        history: int, the number of time lags to use as additional regressors. history = 1 does not use any history data.

        cv: int; specifies number of data splits for cross-validation

        scale_auc: bool; Sets ROC AUC values below 0.5 to 0 and scales the remaining values to an index between 0 and 1.

        remove_nontarget: bool; specifies whether data where non-target behaviours (other than tuned_to) are ocurring is removed
                                from the baseline fluorescence dataset, since keeping it can cause tuning to the target to be underestimated

        subtract_shuffled_tuning: bool; Specifies if the tuning index obtained from shuffled calcium data should be subtracted
                                        from the tuning index estimated from true data.

    Returns:
        aucs: an array of ROC AUC values for each cell
    '''

    # Find timepoints when non-target behaviours are ocurring so they can be excluded from the baseline
    if remove_nontarget:

        exceptions = []

        if tuned_to in ('push', 'retreat', 'resist'):
            exceptions = ['tube test']

        if tuned_to in ('sniff', 'sniff AG'):
            exceptions = ['chemoinvestigation']

        if tuned_to == 'chemoinvestigation':
            exceptions = ['sniff', 'sniff AG']

        for exception in exceptions:
            if not exception in binary.columns:
                exceptions.remove(exception)

        exceptions.append(tuned_to)

        included = ~binary[binary.columns.drop(exceptions)].sum(axis=1).astype(bool)
    else:
        included = [True] * df.shape[1]

    # Find the binary array for the desired behaviour and remove indices where non-target behaviours are ocurring
    tuned_to = binary[tuned_to][included]

    # Define model
    model = LogisticRegression(solver='saga', penalty='l2', max_iter=2500)

    # Prepare empty result list
    aucs = []

    # Loop over cells in the recording
    for cell in df:

        # Remove cell activity data where non-target behaviours are ocurring
        cell = cell[included]

        # Make design matrix
        X = make_design_matrix(cell, history)

        # 8-fold cross-validate the model on data
        model_score = cross_val_score(model, X, tuned_to, scoring='roc_auc', cv=cv).mean()

        # Scale AUC scores to interval from 0 to 1
        if scale_auc:

            model_score = (model_score - 0.5) * 2
            if model_score < 0: model_score = 0

        # Estimate tuning to a shuffled calcium dataset
        if subtract_shuffled_tuning:

            shuffled_cell = cell.copy()
            np.random.shuffle(shuffled_cell)

            shuffled_X = make_design_matrix(shuffled_cell, history)

            shuffled_score = cross_val_score(model, shuffled_X, tuned_to, scoring='roc_auc', cv=cv).mean()

            # Scale AUC scores to interval from 0 to 1
            if scale_auc:

                shuffled_score = (shuffled_score - 0.5) * 2
                if shuffled_score < 0: shuffled_score = 0

            model_score -= shuffled_score

            if model_score < 0: model_score = 0

        # Determine if tuning is positive or negative and represent that in sign of the AUC score
        if cell[tuned_to != 0].mean() < cell[tuned_to == 0].mean():
            model_score *= -1

        # Append result to return variable
        aucs.append(model_score)

    aucs = np.array(aucs)

    return aucs


def episode_based_tuning(episodes, event_index=None, history=1, cv=8):
    '''
    Calculates cell behavioural tuning based on behavioural episode data - temporal windows around a behavioural event.
    Uses a logistic GLM to model the change in activity from baseline state to behavioural state.

    Args:
        episodes: An IxJxK matrix, where I are episodes, J are cells and K are timepoints in the behavioural episode
        event_index: int; Index in a behavioural episode window which separates
        history: int, the number of time lags to use as additional regressors. history = 1 does not use any history data.
        cv: int, specifies fold crossvalidation.

    Returns:
        aucs: an array of ROC AUC values for each cell
    '''

    # Define model
    model = LogisticRegression(solver='saga', penalty='l2', max_iter=2500)

    if event_index is None:
        event_index = episodes.shape[2] // 2

    # Pool baseline and signal data across different episodes
    baseline = episodes[:, :, :event_index]
    baseline = np.concatenate(baseline, axis=1)
    signal = episodes[:, :, event_index:]
    signal = np.concatenate(signal, axis=1)

    # Define the encoding variable which splits each episode into the baseline and signal (behaviour) dataset.
    index = np.concatenate((np.zeros(baseline.shape[1]), np.ones(signal.shape[1])))
    index = index.astype(bool)

    def normalise_auc(auc):
        auc = (auc - 0.5) * 2
        if auc < 0:
            auc = 0

        return auc

    # Prepare empty output variable
    aucs = np.zeros(episodes.shape[1]) * np.nan

    for j, cell in enumerate(zip(baseline, signal)):

        cell = np.concatenate((cell[0], cell[1]))

        # Reformat data into design matrix and create a shuffled calcium dataset
        X = make_design_matrix(cell, history=history)

        shuffled_cell = cell.copy()
        np.random.shuffle(shuffled_cell)
        X_shuffled = make_design_matrix(shuffled_cell, history=history)

        # 8-fold cross-validate model on real and shuffled data
        model_score = cross_val_score(model, X, index, scoring='roc_auc', cv=cv).mean()
        shuffled_score = cross_val_score(model, X_shuffled, index, scoring='roc_auc', cv=cv).mean()

        # Scales aucs to interval between 0 and 1. Scores below chance rate are set to 0.
        model_score = normalise_auc(model_score)
        shuffled_score = normalise_auc(shuffled_score)

        # Subtract shuffled score from actual score
        model_score -= shuffled_score
        if model_score < 0:
            model_score = 0

        # Determine if tuning is positive or negative and represent that in sign of the AUC score
        if cell[index].mean() < cell[~index].mean():
            model_score *= -1

        aucs[j] = model_score

    return aucs


def tuning_barplot(tuning_scores, color='C0', ax=None):
    if ax is None:
        ax = plt.gca()

    ## Set axis properties
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, len(tuning_scores))
    plt.gca().invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.barh(np.arange(0.5, len(tuning_scores) + 0.5, 1), tuning_scores, 0, color=color)
    ax.axvline(0, c='black', linestyle='--')

    ax.set_xticks(np.arange(-1, 1, 0.5))
    ax.set_xlabel('Tuning index: ' + tuning_scores.name)

    return ax


def plot_data(df, behavs, events, tuning_scores, title, fr=20):
    '''

    Plots analysed Inscopix data with behaviour.

    Args
        df: A 2D array containing the calcium data of shape NxM, where N are cells and M are samples.

        behavs: A 1D array of the names of all behaviours in the recording

        events: A list of event arrays for each behaviour (used for the ethogram)

        tuning_scores: a Pandas Series (with a name attribute) of behavioural tuning scores for each cell
                      (must be the same length as number of rows in df)

        title: str; The title of the figure

        fr: int; fps of the calcium recording
    '''

    plt.rcParams.update({'font.size': 18,
                         'lines.linewidth': 2})

    # Setting up figure
    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1.5, 5], 'width_ratios': [5, 1]}, figsize=[20, 7])
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.01,
                        wspace=0.05)

    # Plotting the behaviour raster plot
    colors = ['C{}'.format(i) for i in range(len(events))]
    axs[0, 0].eventplot(events, colors=colors, linelengths=0.8)
    axs[0, 0].set_title(title)

    axs[0, 0].set_xlim(0, df.shape[1])
    axs[0, 0].set_frame_on(False)
    axs[0, 0].xaxis.set_visible(False)

    axs[0, 0].set_yticks(range(len(behavs)))
    axs[0, 0].yaxis.set_tick_params(length=0)
    axs[0, 0].set_yticklabels(behavs)
    for ytick, color in zip(axs[0, 0].get_yticklabels(), colors):
        ytick.set_color(color)

    # Plotting the calcium data
    axs[1, 0].imshow(df, aspect='auto', vmin=0, vmax=1, cmap='magma')
    axs[1, 0].set_frame_on(False)
    axs[1, 0].set_xticks(np.arange(0, df.shape[1], 60 * fr))
    axs[1, 0].set_xticklabels(np.arange(0, df.shape[1], 60 * fr) // (60 * fr))
    axs[1, 0].set_xlabel('Time (min)')
    axs[1, 0].set_ylabel('Cell #')

    # Removing upper right plot axes
    axs[0, 1].set_axis_off()

    # Plotting tuning estimates
    ## Set axis properties
    axs[1, 1].yaxis.set_visible(False)
    axs[1, 1].set_ylim(0, len(tuning_scores))
    plt.gca().invert_yaxis()
    axs[1, 1].spines['left'].set_visible(False)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)

    ## Barplot
    if len(tuning_scores) != df.shape[0]:
        raise ValueError('The number of tuning scores (%s) does not match the number of cells (%s).' % (
            len(tuning_scores), df.shape[0]))

    axs[1, 1].barh(np.arange(0.5, len(tuning_scores) + 0.5, 1), tuning_scores)
    axs[1, 1].axvline(0, c='black', linestyle='--')

    axs[1, 1].set_xticks(np.arange(-1, 1, 0.5))
    axs[1, 1].set_xlabel('Tuning index: ' + tuning_scores.name)

    plt.show()

    return fig


def paired_data_lineplot(data, conditions, marker_colors, line_color, line_width, label=None, xlim=None, ylim=None, xlabel=None,
                         ylabel=None, ax=None):
    '''
    Args:
        data: N pairs by M groups
        conditions: list; List of experimental condition names for the x-axis (strings)
        marker_colors: dict: keys are groups (as integer numbers), values are colours
        line_color: line color
    Returns:
        ax: plt.axes object
    '''

    if ax is None:
        ax = plt.gca()

    for n, group in enumerate(data.T):
        plt.plot(data.T, color=line_color, alpha=1, label=label, zorder=1)
        ax.scatter(x=[n] * len(group), y=group, color=marker_colors[n], zorder=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(conditions)
    ax.xaxis.set_tick_params(length=0)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax


def test_tuning_distribution_differences(tuning_mouse1, tuning_mouse2, alternative='two-sided'):
    '''

    Tests for differences in the distribution of tuning scores for all (common) behaviours between two mice using the
    Kolmogorov-Smirnov test.

    Args
        tuning_mouse1 & tuning_mouse2: pd.DataFrames containing behaviour labelled tuning scores for each cell
        alternative: str; specifies the alternative hypothesis of the KS test, consult scipy.stats.kstest docs for details

    Returns
        difference: dict; a dictionary of KS p-values for each behaviour type
    '''

    differences = {}
    for behav in tuning_mouse1.columns:
        try:
            _, p_value = stats.kstest(tuning_mouse1[behav], tuning_mouse2[behav], alternative=alternative)
            differences[behav] = p_value
        except:
            continue

    return differences


def tuning_proportions(aucs, untuned_boundary=0, tuned_boundary = 0.75):
    '''

    Args:
        aucs: dict / pd.DataFrame: binary behaviour arrays
        untuned_boundary: float: tuning value between which a neuron is considered untuned

    Returns:
        results: dict; a dictionary of tuples containing tuning proportions for each behaviour.
                        The first, second and third array element contain the proportion of positively tuned,
                        untuned and negatively tuned cells respectively.
    '''

    results = {}
    for behav in aucs.columns:
        # Proportion of positively tuned cells
        positive = len(np.where(aucs[behav] > tuned_boundary)[0]) / aucs.shape[0]

        # Proportion of negatively tuned cells
        negative = len(np.where(aucs[behav] < -tuned_boundary)[0]) / aucs.shape[0]

        # Proportion of untuned cells
        untuned = len(np.where(np.abs(aucs[behav]) < untuned_boundary)[0]) / aucs.shape[0]

        results[behav] = np.array([positive, untuned, negative])

    return results


def analyse_recording(path, ID, state,convolve, plotting=True, custom_behaviour=None, average_estimates=False):
    '''

    Args
        path: str; path to directory with recordings
        cage: str; cage number
        rank: str; 'dom', 'sub'
        state: str; 'baseline', 'rising', 'defeated'
        test: str; 'TT', 'RI_male', 'RI_female'
        plotting: bool; specifies if data should be plotted
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots
        average_estimates: bool, specifies whether tuning estimates are repeated and averaged over several random
                                 subsets of behavioural data. Recommended where the number of behavioural episodes
                                 is low (i.e. less than 10). Can significantly slow down performance.

    Returns
        tuning: pd.DataFrame containing behaviour labelled tuning scores for each cell
    '''

    if path[-1] != '/':
        path += '/'


    # Load inscopix data
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')

    # Load and process behaviour data
    num_episodes, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}.csv', df.shape[1])

    # If multiple types of sniffing annotated, create a merged chemoinvestigation behaviour
    try:
        #bdf['chemoinvestigation'] = bdf['sniff'] + bdf['sniff AG']
        bdf['chemoinvestigation'] = bdf['chemoinvestigation'].where(~(bdf['chemoinvestigation'] > 1), other=1)
    except:
        pass

    # Estimate tuning to all behaviours
    tuning = {}
    for behav in bdf.columns:

        # Skip behaviour if behaviour is invalid or not enough data for 8-fold cross-validation
        if behav == 'invalid':
            continue

        elif bdf[behav].sum() < 8:
            continue

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # If averaging, estimate tuning on 10 subsets of the behavioural data
                if average_estimates:
                    try:
                        samples = np.arange(num_episodes[behav][0], dtype=int)[-10:] + 1
                    except:
                        samples = np.arange(num_episodes[behav][0], dtype=int) + 1

                    aucs = np.zeros((df.shape[0], len(samples)))
                    for n, subset in enumerate(samples):
                        subset = behaviour_subset(bdf[behav], subset)
                        subset = pd.DataFrame(subset, columns=[behav])
                        subset = bdf.copy()[behav] = subset

                        aucs[:, n] = estimate_global_tuning(df, subset, behav, history=1, remove_nontarget=False)

                    aucs = aucs.mean(axis=1)

                else:
                    aucs = estimate_global_tuning(df, bdf, behav, history=1, remove_nontarget=False)

                # Check if valid score was produced
                if np.isnan(aucs).sum() > 0:
                    continue
                else:
                    tuning[behav] = aucs

    tuning = pd.DataFrame(tuning)

    # Plotting
    if plotting:
        # If no custom behaviour, find the most strongly tuned behaviour
        if custom_behaviour is None:
            strongest = tuning.columns[tuning.abs().mean().argmax()]

        else:
            strongest = custom_behaviour

        # Sort cells based on tuning similarity to the most strongly tuned behaviour
        sort_index = np.flip(np.argsort(tuning[strongest]))
        tuning.sort_values(by=strongest, ascending=False, inplace=True)
        df = df[sort_index]

        # Convert recording identifiers to presentable text
        #presentable_ranks = {'dom': 'Dominant',
        #                     'sub': 'Subordinate'}

        #presentable_tests = {'TT': 'Tube test',
        #                     'RI_male': 'Male intruder',
        #                     'RI_female': 'Female intruder'}

        #title = presentable_ranks[rank] + ' - ' + presentable_tests[test]

        fig = plot_data(df, bdf.columns, events, tuning[strongest], state + test + custom_behaviour)

    return tuning


def analyse_recording_episodewise(path, state, ID, convolve, window, fr=20, num_episodes=None,
                                  custom_behaviour=None, average_estimates=True, plotting=True,
                                  skip_overlap = False):
    '''

    Args:
        path: str; path to directory with recordings. Required directory structure below:
                    path ---> Behaviour/Inscopix/Behaviour.csv
                      |
                      |-----> Inscopix/Calcium.csv
        cage: str; cage number
        rank: str; 'dom', 'sub'
        state: str; 'baseline', 'rising', 'defeated'
        test: str; 'TT', 'RI_male', 'RI_female'
        window: float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                        signal period after the behavioural event
        fr: int; calcium recording frame rate
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots
        average_estimates: bool; Currently unused
        plotting: bool; specifies if data should be plotted

    Returns:
        tuning: pd.DataFrame of tuning values where rows are cells and columns are behaviours.
    '''

    # Load calcium and behavioural data
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])

    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]

    if skip_overlap:
        bdf = skip_behav_overlap(bdf, window, fr)

    # Merges different types of sniffing into a 'chemoinvestigation' behaviour
    try:
        bdf['chemoinvestigation'] = bdf['chemoinvestigation'].where(~(bdf['chemoinvestigation'] > 1), other=1)
    except:
        pass

    # Estimate tuning for different behaviours
    tuning = {}
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episodes = extract_behav_episodes(df, bdf, behav, window)

        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episodes.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in {ID}')
                    continue
                episodes = episodes[:sample]
            else:
                warn(f'Number of episodes used for tuning estimation is not specified for {behav}.')
                continue

        episodes = behaviour_episode_zscore(episodes, [0, window * fr])
        ##keep record, delete if necessary (edited by MC 27/04/22)
        episodes = np.nan_to_num(episodes)

        for i in range(episodes.shape[0]):
                episodes[i] = minmax_scale(episodes[i], axis = 1)

        aucs = episode_based_tuning(episodes)

        # Check if valid score was produced
        if np.isnan(aucs).sum() > 0:
            continue
        else:
            tuning[behav] = aucs

        tuning[behav] = aucs

    tuning = pd.DataFrame(tuning)

    # Plotting
    if plotting:
        if tuning.empty:

            return tuning

        for behav in bdf.columns:

            strongest = behav

            # Sort cells based on tuning similarity to the most strongly tuned behaviour
            sort_index = np.flip(np.argsort(tuning[strongest]))
            tuning.sort_values(by=strongest, ascending=False, inplace=True)
            df = df[sort_index]


            fig = plot_data(df, bdf.columns, events, tuning[strongest], f'{state}')

    return tuning


## Function adopted by MC in GRIN analysis

## from Inscopix Prelim Analysis
def extract_bdf_avg(path, state, ID, convolve, window, fr, num_episodes = None, skip_overlap = True):
    '''

    Args:
        path: str; path to directory with recordings. Required directory structure below:
                    path ---> Behaviour/Inscopix/Behaviour.csv
                      |
                      |-----> Inscopix/Calcium.csv
        state: str; 'virgin' or PD18
        ID: str; ID of the animal
        convolve: str; 'non_convolved' or 'deconvolved'
        window: float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                        signal period after the behavioural event
        fr: int; calcium recording frame rate
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
        skip_overlap: bol; if True, behaviour episodes that are too close to each other will be deleted from the bdf
        

    Returns:
        episodes: dictionary giving the averaged zscored activities with behaviour as keys of the dictionary.
    '''
       
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')
    
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])
    
    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]

    episodes = {}
    
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episode = extract_behav_episodes(df, bdf, behav, window, fr = fr)
        
        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episode.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in {ID}')
                    continue
                episode = episode[:sample]

            else:
                warn(f'Number of episodes used for tuning estimation is not specified for {behav}.')
                continue
                      
        episodes[behav] = episode
    
    return episodes


def extract_all_bdf(path, state, ID, convolve, skip_overlap = True):
    
    '''

    Args:
        path: str; path to directory with recordings. Required directory structure below:
                    path ---> Behaviour/Inscopix/Behaviour.csv
                      |
                      |-----> Inscopix/Calcium.csv
        state: str; 'virgin' or PD18
        ID: str; ID of the animal
        convolve: str; 'non_convolved' or 'deconvolved'
        skip_overlap: bol; if True, behaviour episodes that are too close to each other will be deleted from the bdf
        

    Returns:
        activities: pd.DataFrame; a dataframe consisting of activities distribution of the recording for all behaviours (behviour parameter
                    stored in the 'Behaviour' column.
    '''
    
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')
    
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])
    
    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]
    activities = pd.DataFrame(columns = ['Behaviour', 'Activities'])
    for behav in bdf.columns:
        activity = {}
        ones = bdf[behav]
        index = np.where(ones == 1)[0]

        # Extract calcium data surrounding behavioural events
        ones = df[:,index]

        activity['Behaviour'] = behav
        activity['Activities'] = [ones]
        activity = pd.DataFrame(activity)
        activities = pd.concat([activities, activity])
    
    return activities
        
    
def population_activities(path, state, ID, convolve,window, fr=10, num_episodes=None,
                                  custom_behaviour=None, skip_overlap=True, z_scored = True):
    
    '''
     
    Zscore calcium imaging data based on the behaviour event and 
    average the zsocred activities across multiple behaviour events.

    Args:

        path: path: str; path to directory with recordings. Required directory structure below:
        state: str; 'virgin', 'second virgin', 'PD18' or 'PD50'
        ID: str； ID of the recorded animal
        convolve: str; 'deconvolved' or 'non_convolved' indicating whether the output calcium traces have gone through deconvolution
        window:float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                        signal period after the behavioural event
        fr:int; fps of the calcium recording
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots

    Returns:

        avg_episodes: a dataframe with structure number of cells * (2* window*fr), 
        containing zscored data averaged across the number indicated by num_episodes


    '''
    
    # Load calcium and behavioural data
    
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation= 'fixed interval')
    
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])
    
    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]
    bdf.to_csv('original.csv')
    
    if skip_overlap:
        bdf = skip_behav_overlap(bdf, window, fr)
        bdf.to_csv('overlapped.csv')

    avg_episodes = {}
    
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episodes = extract_behav_episodes(df, bdf, behav, window)
        
        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episodes.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in {ID}')
                    continue
                episodes = episodes[:sample]

            else:
                warn(f'Number of episodes used for tuning estimation is not specified for {behav}.')
                continue
        
        if z_scored:
            episodes = behaviour_episode_zscore(episodes, [0, window * fr])

        avg_episodes[behav] = np.average(episodes, axis = 0)

    return avg_episodes


def skip_behav_overlap(bdf, window, fr):
    
    '''
     
    identify overlapped behaviour episodes within the identified window and delete the latter of the overlapped
    episode pairs

    Args:

        bdf: a binary pandas dataframe with 0 specifiying absence of a behaviour and 1 specifies presencr of 
             a behaviour
       window: around each behavioural event in seconds that is used for analysis
              e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                signal period after the behavioural event
      

    Returns:

        bdf: a binary pandas dataframe with overlapped behaviour deleted

    '''
    bdf.to_csv('original.csv')
    indices = pd.DataFrame(columns = ['start', 'stop', 'original_stop', 'behav'])
    index = {}
    behavs = list(bdf.columns)
    remove_behav = ['nest time', 'male intruder contact', 'female intruder contact']
    
    behavs = [x for x in behavs if (x not in remove_behav)]
    
    # extract start and stop indices for all behaviour events
    for behav in behavs:
     
        start = np.where((bdf[behav].diff() > 0))[0] 
        stop = np.where(bdf[behav].diff() < 0)[0] 
        index['start'] = [start]
    # obtain stop indices for all behaviour episode other than the ones that associate with the start index 
        index['stop'] = [stop[1:]]
        index['original_stop'] = [stop]
        index['behav'] = [behav]
        index = pd.DataFrame(index)
        
    # save the original start and stop indices pair in numpy array indices
        indices = pd.concat([indices, index])
        
    # obtain start and stop indices pair for every behaviour combination
    indices_ep = indices.explode('start')
    indices_ep = indices_ep.explode('stop')
    
    # obtain the time difference between the start and stop indices
    for i in range(len(indices_ep.start.unique())):
        start = indices_ep.start.unique()[i]
        
        for j in range(len(indices_ep.stop.unique())):
            stop = indices_ep.stop.unique()[j]
            
            diff = start - stop
            
     # erase behaviour only if the interval is shorter than window *fr   
            if diff < window*fr and diff >0:
                deleted = indices_ep[indices_ep.start == start]
            
                deleted_start = int(deleted['start'].unique()[0])
                
                behaviour = deleted['behav'].unique()[0]
                
    # obtain stop index of the behaviour needed to be erased from the original indices array  
                
                deleted_index = indices[indices.behav == behaviour].start[0]
                
                deleted_index = np.where(deleted_index == deleted_start)[0]
                
                deleted_stop = indices[indices.behav == behaviour].original_stop[0][deleted_index]

                bdf.loc[int(deleted_start):int(deleted_stop),behaviour] = 0
                
    bdf.to_csv('deleted.csv')        
    return bdf
    


         
def plot_zscore(df, title, fr=20, clustering = False):
    '''

    Plots analysed averaged zsocred Inscopix data.

    Args
        df: A 2D array containing the calcium data of shape NxM, where N are cells and M are samples.

        title: str; The title of the figure

        fr: int; fps of the calcium recording
        
    '''
    df = pd.DataFrame(df)
    
    if clustering:
        df = hierarchical_clustering(df)
        
    else:
        # Reorder input numpy array based on the index with maximal differential calcium activities

        diff = np.diff(df)
        df['idxsort'] = diff.argmax(axis = 1)  + 1
        df = df.sort_values('idxsort', ascending = True)
        idx = df.index
        df = df.drop(columns=['idxsort'])
    plt.rcParams.update({'font.size': 18,
                         'lines.linewidth': 2})

    # Setting up figure
    fig, ax = plt.subplots(figsize=[5, 5])
    fig.tight_layout()


    # Plotting the calcium data
    ax.imshow(df, aspect='auto',vmin =0, vmax = 3, cmap='viridis')    
    ax.set_frame_on(False)
    ax.set_xticks(np.arange(0, df.shape[1]+2,  fr))
    ax.set_xticklabels(np.arange(int(-df.shape[1]/2), int((df.shape[1]+2)/2), fr) // ( fr))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cell #')
    
    ## set up heatmap legend
    
def plot_amplitude(df, title, cmap, fr=20,clustering = False):
    '''

    Plots analysed averaged zsocred Inscopix data.Recordings are sorted hierarchically
    and each hierarchically determined clusters is sorted based on their response mean 

    Args
        df: A 2D array containing the calcium data of shape NxM, where N are cells and M are samples.

        title: str; The title of the figure

        fr: int; fps of the calcium recording
        
        clustering: bol; True if hierarchical clustering is desired
        
    '''
    df = pd.DataFrame(df)
    
    if clustering:
        df_recorder, clusters = hierarchical_clustering(df)
        df['clusters'] = clusters
        
        df['mean'] = df.iloc[:, -int(df.shape[1]/2):].mean(axis = 1)
        diff = np.diff(df)
        df['idxsort'] = diff.argmax(axis = 1)  + 1
        
        grouped = df.groupby(['clusters'], as_index = False)['idxsort'].mean()
        df = df.drop(columns = ['idxsort'])
        
        df = df.merge(grouped, how = 'outer', on = ['clusters'])
        df = df.sort_values(['mean', 'idxsort'], ascending = False)
        idx = df.index
        df = df.drop(columns = ['idxsort', 'clusters'])
        
        
    else:
        # Reorder input numpy array based on the index with maximal differential calcium activities
        df['mean'] = df.iloc[:, -int(df.shape[1]/2):].mean(axis = 1)
        diff = np.diff(df)
        df['idxsort'] = diff.argmax(axis = 1)  + 1
        df = df.sort_values(['mean','idxsort'], ascending = False)
        idx = df.index
        df = df.drop(columns=['idxsort', 'mean'])
        
        
    plt.rcParams.update({'font.size': 18,
                         'lines.linewidth': 2})

    # Setting up figure
    sns.set(font_scale=2) 
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=[3, 4])
    fig.tight_layout()
    plt.title(title, pad=25)
    plt.xticks()
    plt.yticks()

    # Plotting the calcium data
    
    # cmap=LinearSegmentedColormap.from_list('rg',["green", "w", "orange"], N=256) 
    
    ax.imshow(df, aspect='auto',vmin =-10, vmax = 10, cmap=cmap)
    ax.set_frame_on(False)
    ax.set_xticks(np.arange(0, df.shape[1]+2,  fr))
    ax.set_xticklabels(np.arange(int(-df.shape[1]/2), int((df.shape[1]+2)/2), fr) // ( fr))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Time (s)', labelpad=20)
    ax.set_ylabel('Neuron ID')
    
    heatmap = plt.pcolor(df, cmap = cmap, vmin = -10, vmax = 10)
    plt.colorbar(heatmap)
   
    plt.show()
    
    return fig

def hierarchical_clustering(df):
    
    '''

    hierarchically cluster the Inscopix data on the neuron axis.

    Args
        df: A 2D array containing the calcium data of shape NxM, where N are cells and M are samples.

    Return
        df_recorder: np array; reordered inscopix calcium data
        clusters: a 1d array indicating the identity of the cluster each individual neuron belong to
    '''
    
    # heatmap_sns = sns.clustermap(df, metric="correlation", standard_scale=1, 
    #                              method="average", cmap="bwr",col_cluster=False)
    
    heatmap_sns = sns.clustermap(df, metric="euclidean", standard_scale=1, 
                                 method="ward", cmap="bwr",col_cluster=False)
    
    
    df_reorder = heatmap_sns.data2d  
    
    # get recorded row indicied 
    rowname_list = [df.index[row_id] for row_id in heatmap_sns.dendrogram_row.reordered_ind]
    
    # update dataframe based on row but not column
    df_reorder = df.reindex(rowname_list)
    
    # get cluster id of each nruon
    d = sch.distance.pdist(df)
    L = sch.linkage(d, method='complete')
    # 0.2 can be modified to retrieve more stringent or relaxed clusters
    clusters = sch.fcluster(L, 0.1*d.max(), 'distance')

    return df_reorder, clusters

def plot_ethogram(time, events,behavs):
    '''

    plot ethogram.

    Args
        time: float; indicating the length of the behaviour recording
        events: a binary multidimensional numpy array indicating the presence and absence of each individual behaviour
        behavs: list; list of unique behaviours

    '''

    fig, ax = plt.subplots(figsize=[20, 7])
    fig.tight_layout()
    
    # Plotting the behaviour raster plot
    colors = ['C{}'.format(i) for i in range(len(events))]
    ax.eventplot(events, linelengths=0.8)
    

    ax.set_xlim(0, time)
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)

    ax.set_yticks(range(len(behavs)))
    ax.yaxis.set_tick_params(length=0)
    ax.set_yticklabels(behavs)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)


def cdf(data):
    '''

    compute the cumulative distribution function of a given distribution

    Args
        data: numpy array 
    '''
    
    count, bins_count = np.histogram(data,range = [0,1], bins = 50)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf,bins_count


## from population feature analysis

def population_z_score(path, state, ID, convolve,window, fr=10, num_episodes=None,
                                  custom_behaviour=None, skip_overlap=True):
    
    '''
     
    Zscore calcium imaging data based on the behaviour event and 
    average the zsocred activities across multiple behaviour events.

    Args:

        path: path: str; path to directory with recordings. Required directory structure below:
        state: str; 'virgin', 'second virgin', 'PD18' or 'PD50'
        ID: str； ID of the recorded animal
        convolve: str; 'deconvolved' or 'non_convolved' indicating whether the output calcium traces have gone through deconvolution
        window:float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                        signal period after the behavioural event
        fr:int; fps of the calcium recording
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots

    Returns:

        avg_episodes: a dataframe with structure number of cells * (2* window*fr), 
        containing zscored data averaged across the number indicated by num_episodes


    '''
    
    # Load calcium and behavioural data
    
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')
    
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])
    
    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]
    bdf.to_csv('original.csv')
    
    if skip_overlap:
        bdf = skip_behav_overlap(bdf, window, fr)
        bdf.to_csv('overlapped.csv')

    avg_episodes = {}
    
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episodes = extract_behav_episodes(df, bdf, behav, window)
        
        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episodes.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in {ID}')
                    continue
                episodes = episodes[:sample]

            else:
                warn(f'Number of episodes used for tuning estimation is not specified for {behav}.')
                continue
                
        episodes = behaviour_episode_zscore(episodes, [0, window * fr])

        avg_episodes[behav] = np.average(episodes, axis = 0)

    return avg_episodes



def condition_pairplot(data, id_list, conditions):
    
    num_plot = data[data.IDs.isin(id_list)]
    num_v = np.array(num_plot[num_plot.State == conditions[0]].num_neuron)
    num_P = np.array(num_plot[num_plot.State == conditions[1]].num_neuron)
    num = np.vstack([num_v,num_P]).T

    plt.subplots(figsize = [1,2.5])
    paired_data_lineplot(num, conditions = conditions,
                            marker_colors = ['#0000ff', '#d7301f'],
                            line_color = 'grey', line_width = 2)

    plt.xticks(rotation=45)
    plt.yticks()
    plt.ylabel('Detected neurons', labelpad=5)
    plt.margins(x=0.2)
    sns.despine(bottom=True)
    plt.ylim(0, 60)

def cdf(data):
    count, bins_count = np.histogram(data,range = [-1,1])
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf,bins_count


def lifetime_sparseness(responses):
    # sparseness = 1-(sum of mean responses to all sensory stimuli / N)squared / (sum of (squared mean responses / N)) / (1-(1/N))
    # N = number of investigation behaviour
    # after Vinje & Gallant, 2000; Froudarakis et al., 2014
    N = float(len(responses))
    ls = ((1-(1/N) * ((np.power(responses.sum(axis=1),2)) / (np.power(responses,2).sum(axis=1)))) / (1-(1/N)))
    return ls

def population_sparseness(responses):
    # sparseness = 1-(sum of mean responses for all neurons / N)squared / (sum of (squared mean responses for all nerons / n)) / (1-(1/N))
    # N = number of neurons
    # after Vinje & Gallant, 2000; Froudarakis et al., 2014
    N = float(len(responses))
    ps = ((1-(1/N) * ((np.power(responses.sum(axis=0),2)) / (np.power(responses,2).sum(axis=0)))) / (1-(1/N)))
    return ps

def heatmap(data, node_color, fr = 10):
    fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [5, 1]},figsize=[20, 6])
    cmap=LinearSegmentedColormap.from_list('rg',["b", "w", "r"], N=256) 
    
    # plot calcium traces
    axs[0].imshow(data.T, aspect='auto',vmin =0, vmax = 1, cmap=cmap)    
    axs[0].set_frame_on(False)
    #axs[0].set_xticks(np.arange(0, data.shape[0]+2,  fr))
    #axs[0].set_xticklabels(np.arange(0, (data.shape[0]+2), fr) // ( fr))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Cell #')
    plt.show()
    
    
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


### from multivariate correlation notebook
def calc_MI(x, y, bins, state, mouse, plotting = False):
    c_xy = np.histogram2d(x, y, bins)[0]
    if plotting:
        plt.hist2d(x, y, bins=bins)
        plt.title(state + mouse)
        plt.show()
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

### from Tuning analysis notebook

def extract_episodes(path, state, ID, convolve,window, fr=20, num_episodes=None,
                                  custom_behaviour=None, skip_overlap=True):
    
    '''
     
    extract behaviour episode based on num_episodes and skip_overlap request
    
    Args:

        path: path: str; path to directory with recordings. Required directory structure below:
        state: str; 'virgin', 'second virgin', 'PD18' or 'PD50'
        ID: str； ID of the recorded animal
        convolve: str; 'deconvolved' or 'non_convolved' indicating whether the output calcium traces have gone through deconvolution
        window:float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3 second baseline period before the event and a 3 second
                        signal period after the behavioural event
        fr:int; fps of the calcium recording
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots

    Returns:

        avg_episodes: a dataframe with structure number of cells * (2* window*fr), 
        containing zscored data averaged across the number indicated by num_episodes


    '''
    
    # Load calcium and behavioural data
    
    df = load_inscopix(path + f'{ID}/{convolve}/{ID}_registered_{state}'+'.csv', normalisation='fixed interval')
    
    _, bdf, events = preprocess_behaviour(path + f'{ID}/{ID}_behav_{state}_post.csv', df.shape[1])
    
    # Keep only valid behaviours
    behavs = bdf.columns[bdf.columns != 'invalid']
    bdf = bdf[behavs]
    if skip_overlap:
        bdf = skip_behav_overlap(bdf, window, fr)
    
    # Merges different types of sniffing into a 'chemoinvestigation' behaviour
    try:
        bdf['chemoinvestigation'] = bdf['chemoinvestigation'].where(~(bdf['chemoinvestigation'] > 1), other=1)
    except:
        pass

    out_episode = {}
    out_episodes = pd.DataFrame(columns = ['Behaviour', 'Activities'])
    
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episodes = extract_behav_episodes(df, bdf, behav, window, fr = 20)
       
        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episodes.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in {ID}')
                    continue
                episodes = episodes[:sample]

            else:
                warn(f'Number of episodes used for tuning estimation is not specified for {behav}')
                continue
                
        #for i in range(episodes.shape[0]):
        #    episodes[i] = NormalizeData(episodes[i])
            
        out_episode['Behaviour'] = [behav]
        out_episode['Activities'] = [episodes]
        out_episode = pd.DataFrame(out_episode)
        out_episodes = pd.concat([out_episodes, out_episode])

    
    return out_episodes


def pdf(data):
    count, bins_count = np.histogram(data,range = [-1,1], bins=20)
    pdf = count / sum(count)
    
    return pdf,bins_count

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def trace(data, order, ax):
    if order == 'first':
        base = data.loc[:,'first_baseline'].explode()
        print(base)
        act = data.loc[:,'first_activities']
        data = np.concatenate([base,act])
        #data = data.loc['first_activities'].explode().reset_index()
        #g = ax.plot(data, color = 'black')
        print(data.shape)
        sns.heatmap(data)
        
    else:
        base = data.loc[:,'second_baseline'].explode().reset_index()['second_baseline']
        act = data.loc[:,'second_activities'].explode().reset_index()['second_activities']
        data = np.concatenate([base,act])
        # data = data.loc['second_activities'].explode().reset_index()
        #g = ax.plot(data, color = 'black')
        
    #ax.axis('off')
    ax.set_ylim([0,10])
    #ax.set_title(subject + ' ' + beh)
    
def trace_alignment(d1):
    fig,(ax, ax2) = plt.subplots(1, 2, figsize = (3, 1))
    trace(d1, 'first', ax)
    trace(d1, 'second', ax2)
    
