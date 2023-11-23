for dset in celebdf dfdc dfdc_preview openforensics 
do

    # single model binary detection
    for i in {1..5}
    do
        python -m experiments.single_model.evaluate -d $dset -g 1.3 -bs 512 -mn m_detection_${i} -ot detect_fake -O cross_dataset_detection $1
    done

    # single model multiclass detection
    for i in {1..5}
    do
        python -m experiments.single_model.evaluate -d $dset -g 1.3 -bs 512 -mn m_attribution_${i} -ot detect_fake -O cross_dataset_detection $1
    done 
    
    # ensemble binary detection
    python -m experiments.ensemble_avgpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_detection_1 m_detection_2 m_detection_3 m_detection_4 m_detection_5 -O cross_dataset_detection $1

    # ensemble multiclass detection
    python -m experiments.ensemble_avgpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_attribution_1 m_attribution_2 m_attribution_3 m_attribution_4 m_attribution_5 -O cross_dataset_detection $1

    # ensemble one-vs-real detection
    python -m experiments.ensemble_maxpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_onevsreal_1 m_onevsreal_2 m_onevsreal_3 m_onevsreal_4 m_onevsreal_5 -th 0.5 -O cross_dataset_detection $1

    # ensemble one-vs-rest detection
    python -m experiments.ensemble_maxpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_onevsrest_1 m_onevsrest_2 m_onevsrest_3 m_onevsrest_4 m_onevsrest_5 -th 0.5 -O cross_dataset_detection $1

done


