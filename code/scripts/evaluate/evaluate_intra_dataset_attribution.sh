for quality in raw high low
do
    single model multiclass attribution
    for i in {1..5}
    do
        python -m experiments.single_model.evaluate -d ff++${quality} -m test -g 1.3 -bs 512 -ot attribute -mn m_attribution_${i} -O intra_dataset_attribution $1
    done

    # ensemble multiclass attribution
    python -m experiments.ensemble_avgpool.evaluate -d ff++${quality} -m test -g 1.3 -bs 200 -ot attribute -mn m_attribution_1 m_attribution_2 m_attribution_3 m_attribution_4 m_attribution_5 -O intra_dataset_attribution $1
    
    # ensemble one-vs-real attribution
    python -m experiments.ensemble_maxpool.evaluate -d ff++${quality} -m test -g 1.3 -bs 200 -ot attribute -mn m_onevsreal_1 m_onevsreal_2 m_onevsreal_3 m_onevsreal_4 m_onevsreal_5 -th 0.5 -O intra_dataset_attribution $1
    
    # ensemble one-vs-rest attribution
    python -m experiments.ensemble_maxpool.evaluate -d ff++${quality} -m test -g 1.3 -bs 200 -ot attribute -mn m_onevsrest_1 m_onevsrest_2 m_onevsrest_3 m_onevsrest_4 m_onevsrest_5 -th 0.5 -O intra_dataset_attribution $1  

done


