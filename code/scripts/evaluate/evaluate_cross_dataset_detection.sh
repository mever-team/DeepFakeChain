for dset in celebdf dfdc dfdc_preview openforensics 
do
    for i in {1..5}
    do
        python -m experiments.single_model.evaluate -d $dset -g 1.3 -bs 512 -mn m_detection_${i} -ot detect_fake -O cross_dataset_detection $1
    done

    for i in {1..5}
    do
        python -m experiments.single_model.evaluate -d $dset -g 1.3 -bs 512 -mn m_attribution_${i} -ot detect_fake -O cross_dataset_detection $1
    done 
    
    python -m experiments.ensemble_avgpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_detection_1 m_detection_2 m_detection_3 m_detection_4 m_detection_5 -D cuda:0 -th 0.5 -O cross_dataset_detection $1
    python -m experiments.ensemble_avgpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_attribution_1 m_attribution_2 m_attribution_3 m_attribution_4 m_attribution_5 -D cuda:0 -th 0.5 -O cross_dataset_detection $1
    python -m experiments.ensemble_maxpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_onevsreal_1 m_onevsreal_2 m_onevsreal_3 m_onevsreal_4 m_onevsreal_5 -D cuda:0 -th 0.5 -O cross_dataset_detection $1
    python -m experiments.ensemble_maxpool.evaluate -d $dset -g 1.3 -bs 200 -ot detect_fake -mn m_onevsrest_1 m_onevsrest_2 m_onevsrest_3 m_onevsrest_4 m_onevsrest_5 -D cuda:0 -th 0.5 -O cross_dataset_detection $1

done


