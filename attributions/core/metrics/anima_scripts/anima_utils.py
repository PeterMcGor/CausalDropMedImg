import glob
import os
import subprocess
import pandas as pd
import SimpleITK as sitk
from xml.dom import minidom
import tempfile
import time
import numpy as np

import argparse


class SCRIPTS:
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # animaSegPerfAnalyzer is in the same directory as this script
    SEGANALYZER = os.path.join(SCRIPT_DIR, "animaSegPerfAnalyzer")

def get_anima_mesures(inp_label:str, ref_label:str,
                      multi_label:bool=False,
                      maxFalsePositiveRatioModerator:float=0.65,
                      maxFalsePositiveRatio:float=0.7,
                      minOverlapRatio:float=0.1,
                      minLesionVolume:float=3.0,
                      remove_files:bool=False,
                      match_images:bool=False)->dict:
    #TODO add multi_label --> add "-a" to the fucntion call plus reading several files
    #math_images when metadata is notmatched. Be careful normally yourmetada would be really different!!
    #minLesionVolume 3mm^3
    #add -t for threads
    tmp_dir = tempfile.gettempdir()
    xml_output_pre = os.path.join(tmp_dir, time.strftime("%Y%m%d-%H%M%S")+str(np.random.randint(9000000000000000)))
    inp_img_label_path = xml_output_pre+".nii.gz"
    if match_images:
        inp_img_label = sitk.ReadImage(inp_label)
        inp_img_label.CopyInformation(sitk.ReadImage(ref_label))
        sitk.WriteImage(inp_img_label, inp_img_label_path)
        inp_label = inp_img_label_path

    # -X for xml output
    subprocess.run([SCRIPTS.SEGANALYZER, "-i", inp_label, "-r", ref_label, "-d", "-l", "-s", "-o", xml_output_pre,"-X", '-z', str(maxFalsePositiveRatioModerator), '-y',str(maxFalsePositiveRatio), '-x',str(minOverlapRatio), '-v',str(minLesionVolume)])
    xml_output = xml_output_pre+'_global.xml'
    xml_read = minidom.parse(xml_output)
    if remove_files:
        os.remove(xml_output)
        if match_images:
            os.remove(inp_img_label_path)
    return {measure.attributes['name'].value:measure.firstChild.data for measure in xml_read.getElementsByTagName('measure')}

if __name__ == "__main__":
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Process a log file and extract mean validation dice')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('-r', '--references', type=str, required=True, help='Path to the references folder')
    parser.add_argument('-p', '--predictions', type=str, required=True, help='Path to the predictions folder')
    parser.add_argument('-csv', '--csv_ouput', type=str, required=True, help='Path to the predictions folder')
    args = parser.parse_args()

    reference_subjects = glob.glob(os.path.join(args.references, '*.nii.gz'))
    predictions_folder = args.predictions

    results = []
    for ref_subject in reference_subjects:
        subject_name = os.path.basename(ref_subject)
        pred_subject = os.path.join(predictions_folder, subject_name)

        # Skip if prediction doesn't exist
        if not os.path.exists(pred_subject):
            print(f"Warning: No prediction found for {subject_name}")
            continue

        measures = get_anima_mesures(pred_subject, ref_subject, remove_files=True, match_images=False)
        results.append({
            'id': os.path.splitext(subject_name)[0],
            'dataset': args.dataset,
            'ref': os.path.basename(os.path.normpath(args.references)),
            **measures
        })

    # Print out results
    for result in results:
        print(result)
    pd.DataFrame(results).to_csv(args.csv_ouput)



