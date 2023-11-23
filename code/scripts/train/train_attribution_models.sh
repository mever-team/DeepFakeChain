python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot attribute -mn m_attribution_1 -mt efficientnetv1b0 -sd 0 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot attribute -mn m_attribution_2 -mt efficientnetv1b0 -sd 1 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot attribute -mn m_attribution_3 -mt efficientnetv1b0 -sd 2 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot attribute -mn m_attribution_4 -mt efficientnetv1b0 -sd 3 -e $1
python -m experiments.single_model.train -d ff++raw -g 1.3 -bs 55 -a selimsef -ot attribute -mn m_attribution_5 -mt efficientnetv1b0 -sd 4 -e $1
