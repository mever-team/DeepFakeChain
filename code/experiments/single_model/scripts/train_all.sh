BASEDIR=$(dirname "$0")
echo $BASEDIR
$BASEDIR/trainattributionensemble.sh $1 $2 $3
$BASEDIR/traindetectionensemble.sh $1 $2 $3
$BASEDIR/trainonedetection.sh $1 $2 $3
$BASEDIR/trainonevsrest.sh $1 $2 $3




