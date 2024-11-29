import pandas as pd
import numpy as np
import pydicom
from datetime import datetime
import pickle
from pydicom.dataset import Dataset as DcmDataset
from pydicom.tag import BaseTag as DcmTag
from pydicom.multival import MultiValue as DcmMultiValue
from pydicom.datadict import keyword_for_tag
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os, sys, glob, re
from joblib import dump, load
from fastai.basics import delegates
from fastcore.parallel import parallel
from fastcore.utils import gt
from fastcore.foundation import L
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

### local imports ###
from config import file_dict, abd_label_dict, classes, column_lists, feats
from config import val_list, train_val_split_percent, random_seed, data_transforms
from config import sentence_encoder, series_description_column

### gets the dicom files from a provided directory ###
def get_dicoms(path, first_dcm=False, **kwargs):
    "Walk `path` to get DICOM file names from specific extensions, then read files into a `pandas.DataFrame`. If `first_dcm=True`, only read first file from each folder."
    fns = L()
    extension_list=['.dcm','.dicom','.dcim','.ima']
    print("Finding DICOM files. This may take a few minutes.")
    if first_dcm:
        for r, d, f in os.walk(path):
            if f:
                if Path(f[0]).suffix.lower() in extension_list:
                    fns.append(Path(f'{r}/{f[0]}'))
    else:
        fns = L()
        for r, d, fs in os.walk(path):
            for f in fs:
                if Path(f).suffix.lower() in extension_list:
                    fns.append(Path(f'{r}/{f}'))
    print("Reading DICOM files with extensions .dcm, .dicom, .dcim, or .ima. This may take a few minutes, depending on the number of files to read...")
    df = pd.DataFrame.from_dicoms(fns, **kwargs)
    return fns, df

### Reads a DICOM file and returns the corresponding pydicom.Dataset ###
def dcmread(fn: Path, no_pixels=True, force=True):
    return pydicom.dcmread(str(fn), stop_before_pixels=no_pixels, force=force)

def cast_dicom_special(x):
    cls = type(x)
    if not cls.__module__.startswith('pydicom'): return x
    if cls.__base__ == object: return x
    return cls.__base__(x)

def split_elem(res, k, v):
    if not isinstance(v, DcmMultiValue): return
    for i, o in enumerate(v): res[f'{k}{"" if i == 0 else i}'] = o


def as_dict(self: DcmDataset, filt=True, split_multi=False):
    if filt:
        vals = [self[o] for o in self.keys() if self[o].keyword in column_lists['dicom_cols']]
        
    else:
        vals = [self[o] for o in self.keys()]
    items = [(v.keyword, v.value.name) if v.keyword == 'SOPClassUID' else (v.keyword, v.value) for v in vals]
    res = dict(items)
    res['fname'] = self.filename
    if split_multi:
        for k, v in items: split_elem(res, k, v)
        for k in res: res[k] = cast_dicom_special(res[k])
    return res
DcmDataset.as_dict = as_dict


def dcm2dict(fn, excl_private=False, **kwargs):
    ds = dcmread(fn, **kwargs)
    if excl_private: ds.remove_private_tags()
    return ds.as_dict(**kwargs)


@delegates(parallel)
def from_dicoms(cls, fns, n_workers=0, **kwargs):
    return pd.DataFrame(parallel(dcm2dict, fns, n_workers=n_workers, **kwargs))
pd.DataFrame.from_dicoms = classmethod(from_dicoms)


def get_series_fp(fn): return Path(fn).parent

 ### takes the contents of the dataframe column with path/filenames and converts pieces into separate df columns ###   
def expand_filename_into_columns(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[iterator])
    return df

### another version which goes backwards from the end of the filename
def expand_filename(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[-iterator])
    return df

def extract_image_number(filename):
    pattern = r'-([0-9]+)\.dcm$'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return None


def compute_plane(row):
    '''
    Computes the plane of imaging from the direction cosines provided in the `ImageOrientationPatient` field.
    The format of the values in this field is: `[x1, y1, z1, x2, y2, z2]`,
    which correspond to the direction cosines for the first row and column of the image pixel data.
    '''
    planes = ['sag', 'cor', 'ax']
    if 'ImageOrientationPatient1' in row.keys():
        dircos = [v for k, v in row.items() if 'ImageOrientationPatient' in k]
    else:
        dircos = row['ImageOrientationPatient']

        # Handle MultiValue objects by converting them to a list of floats
        if isinstance(dircos, DcmMultiValue):
            dircos = [float(x) for x in dircos]

    # Check if dircos has the expected length
    if not isinstance(dircos, float) and len(dircos) == 6:
        dircos = np.array(dircos).reshape(2, 3)
        pnorm = abs(np.cross(dircos[0], dircos[1]))
        return planes[np.argmax(pnorm)]
    else:
        return 'unknown'

#_re_extra_info = re.compile(r'[<\([].*?[\]\)>]')

def rm_extra_info(t):
    "Remove extraneous info in closures"
    return re.compile(r'[<\([].*?[\]\)>]').sub('', t).strip()


def detect_contrast(row):
    
    #if entry in contrastbolusagent contains a value
    try:
        c = row['ContrastBolusAgent']
        if type(c) == str: return 1
    except KeyError:
        pass
    # heuristic based on series description text
    sd = rm_extra_info(str(row['SeriesDescription']).lower())
    _c = re.compile(r'(\+-?c|post|with|dyn|portal|equilibrium|hepatobiliary|delayed)')
    if _c.search(sd): return 1

    return 0



# for preprocessing for metadata model
def _make_col_binary(df, col):
    s = df[col].isna()
    if any(s):
        df[col] = s.apply(lambda x: 0 if x else 1)
    else:
        targ = df.loc[0, col]
        df[col] = df[col].apply(lambda x: 0 if x == targ else 1)

# for preprocessing for metadata model
def make_binary_cols(df, cols):
    df1 = df.copy()
    for col in cols:
        if col in df.columns:
            _make_col_binary(df1, col)
        else: 
            df1[col]=0
            _make_col_binary(df1,col)
    return df1

# for preprocessing for metadata model
def rescale_cols(df, cols, scaler=None, need_fit_scaler=False):
    df1 = df.copy()
    if not scaler:
        scaler = MinMaxScaler()
        
    if need_fit_scaler:
        df1[cols] = scaler.fit_transform(df1[cols])
        
        with open('../models/metadata_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    else:   
        df1[cols]= scaler.transform(df[cols])
    
    return df1.fillna(0), scaler

# for preprocessing for metadata model
def get_dummies(df, cols=column_lists['dummies'], prefix=column_lists['d_prefixes']):
    df1 = df.copy()
    for i, col in enumerate(cols):
        df1[col] = df1[col].fillna('NONE')
        mlb = MultiLabelBinarizer()
        df1 = df1.join(
            pd.DataFrame(mlb.fit_transform(df1.pop(col)), columns=mlb.classes_).add_prefix(f'{prefix[i]}_')
        )
    return df1

#get labels from a text file in the format of comma deliminted four columns ()
def labels_from_file(label_path, column_names):

    label_df = pd.read_csv(label_path,header=None)
    label_df.columns=column_names

    return label_df

def preprocess(df, scaler=None, need_fit_scaler = False, save_scaler=False, keep= column_lists['keep'], dummies= column_lists['dummies'], d_prefixes= column_lists['d_prefixes'], binarize= column_lists['binarize'], rescale= column_lists['rescale']):
   #Preprocess metadata for Random Forest classifier to predict sequence type
    print("Preprocessing metadata for Random Forest classifier.")
    df1 = exclude_other(df)
    print(f"Have received {df1.shape[0]} entries.")
    
    # Only keep columns that are both in the DataFrame and the 'keep' list
    df1 = df1[[col for col in keep if col in df1.columns]]
    
    
    if 'PixelSpacing' in df1.columns and df1['PixelSpacing'].any:
        df1['PixelSpacing'] = df1['PixelSpacing'].apply(lambda x: x[0])
    
    # Only get dummies for columns that are in the DataFrame
    dummies = [col for col in dummies if col in df1.columns]
    df1 = get_dummies(df1, dummies, d_prefixes)
    
    # Only make binary columns for columns that are in the DataFrame
    binarize = [col for col in binarize if col in df1.columns]
    df1 = make_binary_cols(df1, binarize)
    
    # for everything that is missing from the feature list set to zeros
    for f in feats:
        if f not in df1.columns:
            df1[f] = 0

    rescale_columns = [col for col in rescale if col in df1.columns]

    if scaler:
        try: 
            check_is_fitted(scaler)
            #print('This scaler is alrady fitted.')
        except NotFittedError:
            print('This scaler is not fitted.')
    df1, scaler = rescale_cols(df1, rescale_columns, scaler, need_fit_scaler)
    
    
            
    return df1, scaler

# populates the components of the file path into separate columns in the dataframe
def expand_filename_into_columns(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[iterator])
    return df


# excludes image types that are not part of this evaluation, such as the pelvis
def exclude_other(df):
    if 'BodyPartExamined' not in df.columns: return df
    other = ['SPINE', 'CSPINE', 'PELVIS', 'PROSTATE']
    filt = df.BodyPartExamined.isin(other)
    df1 = df[~filt].copy().reset_index(drop=True)
    filt1 = df1.SeriesDescription.str.contains(r'(cervical|thoracic|lumbar)', case=False, na=False)
    df2 = df1[~filt1].reset_index(drop=True)
    filt2 = df2.SOPClassUID == "MR Image Storage"
    return df2[filt2].reset_index(drop=True)  

# loads a previously pickled dataset including the train and test dataframes
def load_pickled_dataset(train_file, test_file):
  with open(train_file, 'rb') as f:
    train_df = pickle.load(f)
  with open(test_file, 'rb') as g:
    test_df = pickle.load(g)

  return train_df, test_df

# for some of the previous versions of the csv files, the val set was within the train set and there are certain ones trying to keep 
# consistent across the various models
def create_val_dataset_from_csv(train_file, test_file, val=True, val_lists=val_list):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    if val:
        if val_lists:
            val_df = train_df[train_df.patientID.isin(val_lists)]
            train_df = train_df[~train_df.index.isin(val_df.index)] 
        else:
            train_set, val_set = next(GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 42).split(train_df, groups=train_df['patientID']))
            train_df, val_df = train_set, val_set
        return train_df, val_df, test_df

    else: 
        return train_df, test_df

# loading the train, val, and test dataframes from the csv files
def load_datasets_from_csv(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, val_df, test_df


# this function will select the image in the middle of each series of images, so that only a single image from each series is selected for training
# There is one image from each series for each patient
def shorten_df(df, selection_fraction = 0.5):
  df1 = df.copy()
  grouped_df = df.groupby(['patientID', 'series'])
  sorted_df = grouped_df['file_info'].apply(lambda x: x.sort_values())
  
  selected_filename = grouped_df['file_info'].apply(lambda x: x.sort_values().iloc[int(len(x)*selection_fraction)])

  
  selected_filename = selected_filename.reset_index()
  
  # perform merge and deal with duplicate/unnecessary columns
  df1 = df1.merge(selected_filename, on=['patientID', 'series'], how='left') 
  df_short = df1.drop(['file_info_x', 'img_num'], axis=1)
  df_short = df_short.rename(columns = {'file_info_y': 'file_info'})
  df_short.drop_duplicates(inplace=True)
  df_short.reset_index(drop=True, inplace=True)
  return df_short

# like shorten_df but does not adjust the dataframe, just returns the selected filenames
def mask_one_from_series(df, selection_fraction=0.5):
    df1 = df.copy()
    grouped_df = df.groupby(['patientID', 'series'])
    sorted_df = grouped_df['file_info'].apply(lambda x: x.sort_values())
  
    selected_rows = grouped_df['file_info'].apply(lambda x: x.sort_values().iloc[int(len(x)*selection_fraction)])
   
    return selected_rows

# this gets the dicom images from the column of filenames, and adds contrast and plane information
def prepare_df(df):
    df1 = df.copy()
    filenames = df1.file_info.tolist()
    getdicoms = pd.DataFrame.from_dicoms(filenames)
    merged = getdicoms.merge(df1, left_on='fname', right_on='file_info')
    merged.drop(columns=['file_info'], inplace=True)
    
    merged['contrast'] = merged.apply(detect_contrast, axis=1)
    merged['plane'] = merged.apply(compute_plane, axis=1)
    
    
    return merged


        
#visualization of a batch of images
def imshow(img, title):
    img = torchvision.utils.make_grid(img, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)

    plt.axis('off')
    plt.show()

# produce high quality confusion matrix image and save it
# still need to add labels
def plot_and_save_cm(ytrue, ypreds, fname):
    cm = confusion_matrix(ytrue, ypreds)
    plt.figure(figsize=(25, 25))
    plt.tight_layout()
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.savefig(fname + datetime.now().strftime('%Y%m%d') + ".png", dpi=300, bbox_inches='tight')

    plt.show()

    return plt


# adds the preds and probs to select rows from the original data frame (patient and study info)
def make_results_df(preds, probs, true, df):
    return pd.DataFrame({'preds': preds, 'true': true, 'probs': [row.tolist() for row in probs], 'patientID': df['patientID'], 'series_description': df['SeriesDescription'], 'contrast': df['contrast'], 'plane': df['plane']  })


    
def display_and_save_results(y_pred, y_true, classes=classes, fn='', saveflag = True):
   
   # Generate a confusion matrix based on the true labels and predicted labels
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred, labels=classes)
  
    mask = np.all(cm == 0, axis=1)

    class_text_labels = [abd_label_dict[str(x)]['short'] for x in classes]
    class_text_labels = class_text_labels[~mask]

     # Generate a classification report based on the true labels and predicted labels
    print(classification_report(y_true, y_pred))

    
    cm = cm[~mask]
    # Create a ConfusionMatrixDisplay object with the correct labels
    cm_display = ConfusionMatrixDisplay(cm, display_labels=class_text_labels).plot(xticks_rotation = 'vertical', cmap='Blues')
    plt.figure(figsize=(25, 25))
    plt.tight_layout()
    #ConfusionMatrixDisplay(cm, display_labels=class_text_labels).plot(xticks_rotation = 'vertical', cmap='Blues')
    if saveflag:
        plt.savefig("../assets/FigCM_"+fn+datetime.today().strftime('%Y%m%d')+".tif",dpi=300, bbox_inches = 'tight')     

    return cm      

def update_paths(df, old_base_path, new_base_path):
    if old_base_path and new_base_path:
        df['file_info'] = df['file_info'].str.replace(old_base_path, new_base_path)
    return df



def create_datasets(train_datafile, val_datafile, test_datafile, old_base_path=None, new_base_path=None):
    # reads in the dataframes from csv
    train_full = pd.read_csv(train_datafile)
    val_full = pd.read_csv(val_datafile)
    test_full = pd.read_csv(test_datafile)

    # selects the middle image from each series for further evaluation
    train = shorten_df(train_full)
    val = shorten_df(val_full)
    test = shorten_df(test_full)

    # Update the paths
    train = update_paths(train, old_base_path, new_base_path)
    val = update_paths(val, old_base_path, new_base_path)
    test = update_paths(test, old_base_path, new_base_path)

    
    # changes to the dataframe including adding contrast and computed plane columns
    train_df = prepare_df(train)
    val_df = prepare_df(val)
    test_df = prepare_df(test)

    return train_df, val_df, test_df

