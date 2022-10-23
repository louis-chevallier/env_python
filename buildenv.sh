alias testtorch='testtorchf ${BASH_SOURCE##*/}:${LINENO}'
ABS_SOURCE_PATH=$(realpath  ${BASH_SOURCE})
ABS_SOURCE_PATH_DIR=$(dirname   ${ABS_SOURCE_PATH})

TOOLS_=${TOOLS:-/mnt/hd1/tools}

function testtorchf() {
    echo $@
    which python
    python --version
    cat <<EOF > /tmp/f1.py
import sys
import numpy
from utillc import *;

EKOX(sys.version)
try :
    import torch, torchvision; 
    EKOX(torch.__version__)
    EKOX(torch.version.cuda)
    EKOX(torchvision.__version__)
    a=torch.rand(5,3).cuda()
    EKOX(torch.cuda.get_device_properties(0))
    EKOT('so far so good')
except Exception as e :
    EKOX(str(e))
EKOX(numpy.__version__)
try :
    import pytorch3d
    EKOX(pytorch3d.__version__)
    EKOX('all good!')   
except Exception as e :
    EKOX(e);
EKOT('next tensorboard..')
try :
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/tmp/runs')
    print('tensorboard ok')
except :
    print(' pb avec tensorboard')

try :
    import torch_geometric
    EKOX(torch_geometric.__version__)
except Exception as e :
    EKOX(e)
try :
    EKOT("jax")
    import jax.numpy as jnp
    from jax import jit
    from jax.lib import xla_bridge  
    EKOX(xla_bridge.get_backend().platform)
    def slow_f(x):
        # Element-wise ops see a large benefit from fusion
        y = x * x + x * 2.0
        return y
    x = jnp.ones((5000, 5000))
    fast_f = jit(slow_f)
    EKO()
    # ~ 4.5 ms / loop on Titan X
    [fast_f(x) for i in range(10)]
    EKOT('fast')
    # ~ 14.5 ms / loop (also on GPU via JAX) 
    [ slow_f(x) for i in range(10)]
    EKOT('slow')
except Exception as e :
    EKOX(e)
EKOT("other modules..")
import importlib
modules = [ 'skimage', 'networkx', 'PyQt5', 'trimesh', 'cv2', 'imageio', 'matplotlib']
for m in modules : 
    EKON(m)
    mm = importlib.import_module(m)
    try :
       EKOX(mm.__version__)
    except :
       pass
EKOX('success')
EOF

    python /tmp/f1.py
    
}

function inred() {
    red=`tput setaf 1`
    green=`tput setaf 2`
    reset=`tput sgr0`
    bold=`tput bold`    # Select bold mode
    under=`tput smul`    # Select bold mode
    echo "${under}${bold}${red}$*${reset}"
    #red text ${green}green text${reset}"
}

function testXX() {
    inred en rouge
}

function myhome() {
    if [ -d '/home/wp02/' ] ; then
        local ret="home"
    else
        local ret="srv"
    fi
    echo $ret
}


function install_modules() {
        which python
        which pip
        #ls -l ${ANA}/bin/pip
        which pip
        python --version
        #conda update -y --prefix ${ANA}  anaconda
        conda update conda -y
        #conda install -y python=${PYTHON_VERSION_}
        which python
        conda install -c anaconda -y python=${PYTHON_VERSION_}
        python --version
        echo ${ANA}

        if [ 0${WITH_FBX} == 0yes ]; then
           cp /mnt/hd2/tools/fbx202031_fbxpythonsdk_linux/lib/Python37_x64/* ${ANA}/lib/python${PYTHON_VERSION_}/site-packages
           python /mnt/hd2/tools/fbx202031_fbxpythonsdk_linux/samples/ExportScene01/ExportScene01.py
        fi
        
        conda install -y pip

        which pip
        pip install --upgrade pip

        #pip install utillc
        
        if [ 0${WITH_TORCH_} == 0yes ]; then
            #conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-lts -c nvidia
            if [ 0${TORCH_VERSION} == 1.11.0 ]; then
                conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
            else    
                conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
            fi
            pip install torchsummary torchscan
            pip install pytorch-hed
            pip install torchviz # sudo apt install graphviz
            conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath

            if [ 0${PYTORCH3D_VERSION} == 06_2 ]; then
                conda install -y pytorch3d -c pytorch3d
            elif   [ 0${PYTORCH3D_VERSION} == 0nightly ]; then
                conda install -y pytorch3d -c pytorch3d                
            else
                pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
            fi
            
        fi
        #testtorch   || true        
        if [ 0${WITH_JAX_} == 0yes ]; then
            testtorch   || true 
            pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
            testtorch   || true 
        fi
        pip install ply
        conda install -y -c conda-forge numpy==1.22.3
        conda install -y scipy        
        #testtorch   || true       
        #conda install -y pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.6
        #conda install -c fvcore -c iopath -c conda-forge fvcore iopath
        conda install -y -c bottler nvidiacub

	if [ 1 == 1 ]; then
	    #        conda install -y -c conda-forge -c fvcore -c iopath fvcore iopath
	    #        conda install -y -c fvcore -c iopath fvcore iopath
	    #        conda install -y -c pytorch3d pytorch3d


            #conda install -y pytorch3d -c pytorch3d
	    #testtorch  || true       
            conda install -y -c anaconda scikit-learn
            conda install -y matplotlib
	    #testtorch   || true       
            #pip install opencv-python==4.5
            pip install opencv-python scikit-image  imageio plotly
            #matplotlib
            #conda install -y -c conda-forge opencv
	    #=4.5
            #pip install opencv-contrib-python==4.1.2
	    
            pip install loguru
            pip install smplx==0.1.26
            #conda install -y pytorch=1.8.1 torchvision cudatoolkit=${CUDA_VERSION_} -c pytorch
            #conda install -y -c conda-forge nvidia-apex
            # conda install -y  cffi
            # conda install -y cloudpickle  cycler  dask  decorator  imageio  kiwisolver  matplotlib  networkx  numpy  pandas  Pillow  pycparser  pygit  pyparsing  python-dateutil  pytz  PyWavelets  PyYAML  scikit-image  scikit-learn  scipy  six  toolz imageio-ffmpeg  tqdm pandas sympy         
            # pip install tensorboard pycocotools
	    
            #conda install -y -c conda-forge opencv
            #pip install opencv-python
            pip install chumpy

            # #conda install -y sklearn
            pip install albumentations

            if [ 0${WITH_COCO_} == 0yes ]; then            
                pip install pycocotools
            fi
            #pip install optuna
            pip3 install PyQt5

            # requires numba =>  numpy < 1.23
            pip install face-alignment
            pip install ptflops thop

            pip install h5py==3.1.0
	    #testtorch   || true                   
            conda install -y -c conda-forge python-lmdb
            pip install psutil
            pip install utillc
            pip install check-wheel-contents
            pip install twine
            pip install visdom dominate
            pip install moviepy

            #conda install -c conda-forge tianshou
            #apt install swig
            #pip install box2d-py
            if [ 0${WITH_BOX2D_} == 0yes ]; then
                pip install Box2D
                pip install gym[box2d]
            fi
            pip install Cython

            pip install headpose
            pip install mediapipe
            pip install face_detection
            #testtorch   || true
            # rebelote sinon, torch not compiled with cuda .. (?)

            pip install git+https://github.com/louis-chevallier/face-detection.git@master
            pip install bing-image-downloader
            pip install tensorboard

            
            # pyg not compat with python3.10 !!
            #conda install -y pyg -c pyg
            
            conda install -y -c conda-forge trimesh
            conda install -y -c conda-forge rtree
            conda install -y -c conda-forge pyembree
            conda install -y -c conda-forge shapely
            conda install -y -c conda-forge pyglet
            conda install -y sympy
            #testtorch   || true

            conda install -y -c conda-forge pycairo
            conda install -y -c conda-forge python-igraph

            #testtorch   || true

            #for DDD
            pip install PyWavefront
            pip install kornia

            # generation de UV maps
            pip install xatlas


            # optional
            cond install -y  -c conda-forge ninja
            pip install pyrender plotoptix
            #pip install freecad
            #pip install pygltflib           
            #pip install siren-pytorch

            pip install pandoc-eqnos

            conda install -y jupyter
            pip install jupytext

            conda install -y -c conda-forge imagemagick
            pip install moviepy

            # download of data file from gdrive
            pip install gdown

            # streaming de video pour demo web
            pip install av
            pip install aiortc

	fi
}

function buildtheenv666() {

    # install driver
    # sudo apt purge "nvidia*" "libnvidia*"
    # sudo apt install nvidia-driver-510
    #
    echo ${TTOOLS}
    TTOOLS=${TOOLSD:-/mnt/hd1/tools}
    DEST=${DESTINATION:-conda666}
    PYTHON_VERSION_=${PYTHON_VERSION:-3.8.0}
    
    DD1=$( dirname ${BASH_SOURCE[0]}) 
#    echo dd1 $DD1
    DD2=$(readlink -m $DD1)
#    echo dd2 $DD2
    pwd
    DD=${DD2}
    source $DD/buildenv.sh
    ANA=${TTOOLS}/${DEST}
    CUDA_VERSION_=${CUDA_VERSION:-11.6}
    #CUDA_VERSION_=11.1
    WITH_TORCH_=${WITH_TORCH:-yes}
    WITH_JAX_=${WITH_JAX:-no}

    echo python version ${PYTHON_VERSION_} torch: ${WITH_TORCH_} torch_version ${TORCH_VERSION} pytorch3d ${PYTORCH3D_VERSION} jax: ${WITH_JAX_} "\n"

#    which python
    if [ 0"$1" == 0install ]; then
        echo installing
        set -vx
        rm -fr ${ANA}
        (cd ${TTOOLS}; wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
        DIST=${TTOOLS}/Miniconda3-latest-Linux-x86_64.sh
        SOURCE=${DIST}
        bash ${SOURCE} -p ${ANA} -b -f
        PATH=${ANA}/bin:${PATH}

        install_modules
        pip freeze > requirements.txt
        set +vx
	#testtorch   || true
        echo "========================================================================"
        echo "                     ENV BUILT "
        echo "========================================================================"  
    else
        which python
        $*
    fi
    #cuda92
    PATH=${ANA}/bin:${PATH}

    which python
    testtorch   || true
}


function build_python_env_docker() {
    echo source
    source buildenv.sh
    echo $1
    TORCH_VERSION=1.11.0
    PYTORCH3D_VERSION=0.7.0
    TOOLSD=. DESTINATION=conda116_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=11.3  buildtheenv666 $*
}

function test_version() {
    source /mnt/hd2/users/louis/dev/git/cara3/smoke/smoke/photometric_optimization/buildenv.sh
    echo $1
    DESTINATION=conda116_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=11.6  buildtheenv666 $*
    DESTINATION=conda113_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=11.3  buildtheenv666 $*
    DESTINATION=conda110_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=11.0  buildtheenv666 $*
    DESTINATION=conda102_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=10.2  buildtheenv666 $*
    DESTINATION=conda101_${TORCH_VERSION}_${PYTORCH3D_VERSION} CUDA_VERSION=10.1  buildtheenv666 $*
}

function test_version1() {
     TORCH_VERSION=1_11_0 test_version $*
     TORCH_VERSION=1_7_1 test_version $*
}

function test_version2() {
    # avec install ou run en params
     PYTORCH3D_VERSION=6_2  test_version1 $* 
     #PYTORCH3D_VERSION=999 test_version1 $*    
 }

function test_version3() {
    source ${ABS_SOURCE_PATH_DIR}/buildenv.sh
    # avec install ou run en params
    DESTINATION=conda110 PYTORCH3D_VERSION=6_2 TORCH_VERSION=1_11_0 CUDA_VERSION=11.6  buildtheenv666 $* 
}

function env_deploy() {
    # avec install ou run en params
    source buildenv.sh
    TOOLSD=${PWD}/env_dir DESTINATION=condaDDD PYTORCH3D_VERSION=6_2 TORCH_VERSION=1_11_0 CUDA_VERSION=11.6  buildtheenv666 $*
    echo ${PWD}
}


function test_version33() {
    source /mnt/hd2/users/louis/dev/git/cara3/smoke/smoke/photometric_optimization/buildenv.sh
    # avec install ou run en params
    DESTINATION=conda11033 PYTHON_VERSION=3.9  PYTORCH3D_VERSION=6_2 TORCH_VERSION=1_11_0 CUDA_VERSION=11.6  buildtheenv666 $* 
 }

function test_version4() {
    source /mnt/hd2/users/louis/dev/git/cara3/smoke/smoke/photometric_optimization/buildenv.sh
    # avec install ou run en params
    DESTINATION=conda110_X PYTORCH3D_VERSION=nightly PYTHON_VERSION=3.9 TORCH_VERSION=1_11_0 CUDA_VERSION=11.6  buildtheenv666 $* 
 }

function set_env() {
    source ./buildenv.sh
    # avec install ou run en params
    DESTINATION=conda110  PYTORCH3D_VERSION=6_2 TORCH_VERSION=1_11_0 CUDA_VERSION=11.6  buildtheenv666 $* 
 }



function setup_android() {
    #https://www.youtube.com/watch?v=5Lxuu16_28o
    sudo apt install openjdk-11-jdk
    java --version
    sudo apt-add-repository ppa:maarten-fonville/android-studio
    sudo apt update
    sudo apt install android-studio
}
    
