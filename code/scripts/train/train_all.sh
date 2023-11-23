BASEDIR=$(dirname "$0")
echo $BASEDIR
$BASEDIR/train_attribution_models.sh "$1"
$BASEDIR/train_detection_models.sh "$1"
$BASEDIR/train_one_vs_real_models.sh "$1"
$BASEDIR/train_one_vs_rest_models.sh "$1"




