pip install -r requirements.txt
mkdir data/

wget -O bpi12.xes.gz https://data.4tu.nl/ndownloader/files/24027287
gunzip bpi12.xes.gz
mv bpi12.xes data/

wget -O bpi17.xes.gz https://data.4tu.nl/ndownloader/files/24044117
gunzip bpi17.xes.gz
mv bpi17.xes data/

mkdir checkpoints/