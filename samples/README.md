
## getting original data

```
wget https://library.ldraw.org/library/updates/complete.zip
unzip complete.zip ## making ldraw directory
```
## generate parts directory

```
mkdir -p parts
PYTHONPATH=$PYTHONPATH:$(pwd)/.. choreonoid -p gen_parts_data.py
```
