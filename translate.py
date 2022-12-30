from sqlalchemy.sql import select
from sqlalchemy import create_engine, MetaData, Table, update, inspect
dbcon = create_engine(
    'sqlite:////Users/scottsyms/code/HeritageCanada/data/fish/sample2.db')
# dbcon.close_all()

metadata = MetaData()
source = Table(
    'source', metadata, autoload=True, autoload_with=dbcon)

print("Columns: ", metadata.tables['source'].columns.keys())


# Modify English
print("Processing English sourced pages...")

result = dbcon.execute(select(
    [source.c.id, source.c.language]))

selectid = []
[selectid.append(i) for i in result]
print(selectid[0][1])

count = 0
for i in selectid:
    # If language is en
    if i[1] == 'en':
        movelanguage = source.update().values(english='ENGLISH again').where(
            source.c.id == selectid[count][0])
        translatealternate = source.update().values(french='Translation from English').where(
            source.c.id == selectid[count][0])
        translatetospanish = source.update().values(spanish='Translation to Spanish').where(
            source.c.id == selectid[count][0])
    elif i[1] == 'fr':
        movelanguage = source.update().values(french='FRENCH again').where(
            source.c.id == selectid[count][0])
        translatealternate = source.update().values(english='Translation from French').where(
            source.c.id == selectid[count][0])
        translatetospanish = source.update().values(spanish='Translation to Spanish').where(
            source.c.id == selectid[count][0])
    dbcon.execute(movelanguage)
    dbcon.execute(translatealternate)
    dbcon.execute(translatetospanish)
    count += 1



print("Done")
