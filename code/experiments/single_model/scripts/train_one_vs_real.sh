python -m experiments.single_model.train -d ff++raw -c real deepfakes -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_1 -mt $1 -e $2 -sd 0
python -m experiments.single_model.train -d ff++raw -c real faceshifter -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_2 -mt $1 -e $2 -sd 0
python -m experiments.single_model.train -d ff++raw -c real neuraltextures -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_3 -mt $1 -e $2 -sd 0
python -m experiments.single_model.train -d ff++raw -c real faceswap -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_4 -mt $1 -e $2 -sd 0
python -m experiments.single_model.train -d ff++raw -c real face2face -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_5 -mt $1 -e $2 -sd 0
