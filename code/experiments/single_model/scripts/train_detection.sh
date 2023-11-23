python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_detection_1 -mt $1 -e $2 -sd 0
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_detection_2 -mt $1 -e $2 -sd 1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_detection_3 -mt $1 -e $2 -sd 2
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_detection_4 -mt $1 -e $2 -sd 3
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_fake -mn m_detection_5 -mt $1 -e $2 -sd 4
