python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_category 1 -mn m_onevsrest_2 -mt efficientnetv1b0 -sd 0 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_category 2 -mn m_onevsrest_2 -mt efficientnetv1b0 -sd 0 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_category 3 -mn m_onevsrest_3 -mt efficientnetv1b0 -sd 0 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_category 4 -mn m_onevsrest_4 -mt efficientnetv1b0 -sd 0 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot detect_category 5 -mn m_onevsrest_5 -mt efficientnetv1b0 -sd 0 -e $1
