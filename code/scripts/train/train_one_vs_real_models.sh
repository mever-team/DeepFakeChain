python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_1 -mt efficientnetv1b0 -sd 0 -c real deepfakes -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_2 -mt efficientnetv1b0 -sd 0 -c real faceshifter -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_3 -mt efficientnetv1b0 -sd 0 -c real neuraltextures -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_4 -mt efficientnetv1b0 -sd 0 -c real faceswap -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_onevsreal_5 -mt efficientnetv1b0 -sd 0 -c real face2face -e $1
