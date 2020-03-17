#!/bin/bash
#SBATCH --job-name=createLabel
#SBATCH --output=createLabel-%j.out
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH -t 02:00:00
#SBATCH -c 1

export FREESURFER_HOME=/cm/shared/openmind/freesurfer/6.0.0;
export SUBJECTS_DIR=/cm/shared/openmind/freesurfer/6.0.0/subjects;
source $FREESURFER_HOME/SetUpFreeSurfer.sh;

cd /mindhive/evlab/u/gretatu/Desktop/

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_langROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_langROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_allROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_allROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_MDROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_MDROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_DMNROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_DMNROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_auditoryROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_auditoryROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

mri_vol2label --i ./surface_projection_v2/ROIs/"${1}"_visualROIs_"${2}"_surface.nii --l ./surface_projection_v2/labels/"${1}"_visualROIs_"${2}".label --id 1 --surf cvs_avg35_inMNI152 "${3}" inflated

