import pickle
import pandas as pd
from keras.utils import to_categorical
import numpy as np


def toPickle(name, mat):
    filehandler = open(name+".pkl", "wb")
    pickle.dump(mat, filehandler)
    filehandler.close()


def getPickle(name):
    filehandler = open(name, "rb")
    object = pickle.load(filehandler)
    filehandler.close()
    return object


PATH = './OTROS_CJTOS/bookcrossing/'

PREFIJO = 'BOOKC'

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# VALORACIONES
# ---------------------------------------------------------------------------------------------------------------------
valoraciones = pd.read_csv(PATH+'BX-Book-Ratings.csv', sep=';')
# de momento dejo el rating como un problema de regresión
print(valoraciones)
# las valoraciones "0" quieren explesar que el usuario tuvo alguna interacción con el libro, pero no es una nota en sí
# misma, es lo que en su paper llaman valoración implícita. Elimino estas valoraciones para trabajar con las
# valoraciones explícitas únicamente
valoraciones.drop(valoraciones.index[valoraciones.rating == 0], inplace=True)
valoraciones = valoraciones.reset_index(drop=True)

print("usuarios:", valoraciones.user_id.unique().shape)  # (77805,)
print("libros:", valoraciones.isbn.unique().shape)  # (185973,)
print("valoraciones:", valoraciones.shape[0])  # 433671

val_libro = valoraciones.groupby(by=['isbn']).count()
print("libros con al menos 10 valoraciones:", val_libro[val_libro.rating >= 10].shape[0])
idx_libros_10 = val_libro[val_libro.rating >= 10].index
valoraciones = valoraciones[valoraciones['isbn'].isin(idx_libros_10)]

val_user = valoraciones.groupby(by=['user_id']).count()
print("usuarios con al menos 5 valoraciones:", val_user[val_user.rating >= 5].shape[0])
idx_users_5 = val_user[val_user.rating >= 5].index
valoraciones = valoraciones[valoraciones['user_id'].isin(idx_users_5)]

# como se han eliminado las valoraciones de muchos usuarios, ahora no se garantiza que todas las películas
# tengan 10 valoraciones
val_libro = valoraciones.groupby(by=['isbn']).count()
print(val_libro[val_libro.rating >= 10].index.shape)  # 3382
print(val_libro[val_libro.rating >= 5].index.shape)  # 5372
print(val_libro[val_libro.rating >= 1].index.shape)  # 5633
idx_libros_10 = valoraciones.isbn.unique()

print("usuarios:", valoraciones.user_id.unique().shape)  # (6029,)
print("libros:", valoraciones.isbn.unique().shape)  # (5633,)
print("valoraciones:", valoraciones.shape[0])  # 92449

for i in range(1, 10+1):
    print(i, np.sum(valoraciones.rating == i))
# 1 277
# 2 475
# 3 970
# 4 1474
# 5 8469
# 6 6780
# 7 15193
# 8 22879
# 9 16947
# 10 18985

print(">=5:", np.sum(valoraciones.rating >= 5), np.sum(valoraciones.rating >= 5)*100/valoraciones.shape[0])
print(">=6:", np.sum(valoraciones.rating >= 6), np.sum(valoraciones.rating >= 6)*100/valoraciones.shape[0])
print(">=7:", np.sum(valoraciones.rating >= 7), np.sum(valoraciones.rating >= 7)*100/valoraciones.shape[0])
print(">=8:", np.sum(valoraciones.rating >= 8), np.sum(valoraciones.rating >= 8)*100/valoraciones.shape[0])
# >=5: 89253 96.54295882053889
# >=6: 80784 87.38223236595312
# >=7: 74004 80.04845915045051
# >=8: 58811 63.614533418425296
# se podría partir de la siguente manera:
# - rating entre [1, 7] = 0 => 36.39%
# - rating entre [8,10] = 1 => 63.61%
valoraciones = valoraciones.reset_index(drop=True)

# creo un diccionario para renumerar los usuarios válidos del 1 al 6029
ids_usuarios_validos = valoraciones.user_id.unique()
nuevos_ids = list(range(1, ids_usuarios_validos.shape[0]+1))
dic_usuarios = dict(zip(ids_usuarios_validos, nuevos_ids))

# creo un diccionario para renumerar los libros válidos del 1 al 5633
ids_libros_validos = valoraciones.isbn.unique()
nuevos_ids = list(range(1, ids_libros_validos.shape[0]+1))
dic_libros = dict(zip(ids_libros_validos, nuevos_ids))

CREA_VALORACIONES = False
if CREA_VALORACIONES:
    # ahora voy a sustituir los códigos de usuarios y libros para sean número correlativos
    # creo el vector de usuarios actualizado
    us = np.zeros((valoraciones.shape[0],), dtype=np.int64)
    for k in dic_usuarios:
        print(k, dic_usuarios[k])
        idx = valoraciones[valoraciones.user_id == k].index
        us[idx] = dic_usuarios[k]

    # creo el vector de libros actualizado
    li = np.zeros((valoraciones.shape[0],), dtype=np.int64)
    for k in dic_libros:
        print(k, dic_libros[k])
        idx = valoraciones[valoraciones.isbn == k].index
        li[idx] = dic_libros[k]

    valoraciones = pd.DataFrame(list(zip(us, li, valoraciones.rating)), columns=['user_id', 'item_id', 'rating'])
    toPickle(PATH+'valoraciones', valoraciones)
else:
    valoraciones = getPickle(PATH+'valoraciones.pkl')

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ITEMS
# ---------------------------------------------------------------------------------------------------------------------
CREA_LIBROS = False
if CREA_LIBROS:
    # "isbn";"title";"author";"year";"publisher";"imageSmall";"imageMedium";"imageLarge"
    libros = pd.read_csv(PATH+'BX-Books.csv', sep=';')
    print(libros)

    libros = libros[['isbn', 'author', 'year', 'publisher']]

    libros = libros[libros['isbn'].isin(ids_libros_validos)]
    print(libros.shape)              # 5437 descipciones de libros
    print(ids_libros_validos.shape)  # 5633 libros valorados  HAY UNOS 200 LIBROS SIN DESCRIPCIÓN <- OJO OJO OJO OJO
    print("unique(autores):", libros.author.unique().shape)  # 1872
    print("unique(publisher):", libros.publisher.unique().shape)  # 492

    libros = libros.reset_index(drop=True)

    # ahora voy a sustituir los isbn de los libros por los ids del diccionario
    li = np.zeros((libros.shape[0],), dtype=np.int64)
    for i in range(li.shape[0]):
        if i % 100 == 0:
            print(i, dic_libros[libros.loc[i].isbn])
        li[i] = dic_libros[libros.loc[i].isbn]

    libros = pd.DataFrame(list(zip(li, libros.author, libros.year, libros.publisher)),
                          columns=['item_id', 'author', 'year', 'publisher'])
    print(libros)
    toPickle(PATH+'libros', libros)
else:
    libros = getPickle(PATH + 'libros.pkl')

CATEGORIZAR_LIBROS = False
if CATEGORIZAR_LIBROS:
    # cambio autores por números
    autores = libros.author.unique()
    num_autores = autores.shape[0]   # 1872
    dic_autores = dict(zip(autores, list(range(num_autores))))
    # ahora voy a sustituir los autores de los libros por los ids del diccionario
    au = np.zeros((libros.shape[0],), dtype=np.int64)
    for i in range(au.shape[0]):
        if i % 100 == 0:
            print(i, dic_autores[libros.loc[i].author])
        au[i] = dic_autores[libros.loc[i].author]
    # y ahora hago un one-hot de autores y lo añado al df
    oh_autores = to_categorical(au, num_classes=num_autores, dtype='int32')
    libros[autores] = oh_autores

    # lo que hice para autores lo hago ahora para publisher
    editoriales = libros.publisher.unique()
    num_editoriales = editoriales.shape[0]
    dic_editoriales = dict(zip(editoriales, list(range(num_editoriales))))
    # ahora voy a sustituir las editoriales de los libros por los ids del diccionario
    ed = np.zeros((libros.shape[0],), dtype=np.int64)
    for i in range(ed.shape[0]):
        if i % 100 == 0:
            print(i, dic_editoriales[libros.loc[i].publisher])
        ed[i] = dic_editoriales[libros.loc[i].publisher]
    # y ahora hago un one-hot de autores y lo añado al df
    oh_editoriales = to_categorical(ed, num_classes=num_editoriales, dtype='int32')
    libros[editoriales] = oh_editoriales

    # y finalmente elimino las columnas no categorizadas
    libros.drop(['author', 'publisher'], axis='columns', inplace=True)
    toPickle(PATH+'libros_categorizados', libros)
else:
    libros = getPickle(PATH + 'libros_categorizados.pkl')

# hay 2 editoriales y 2 autores con el mismo nombre:
# print(np.intersect1d(autores,editoriales)  # array(['Anatolian Treasures', 'Conari Press'], dtype=object)
# el item_id 5409 tiene como author y publisher: 'Anatolian Treasures'
# el item_id 1807 tiene como author y publisher: 'Conari Press'
# así que la dimensión final de los libros será: [5437 rows x 2364 columns]
# las columnas deberían ser: item_id+year+autores+editoriales=1+1+1872+492=2366 pero son 2364 por la intersección

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# USUARIOS
# ---------------------------------------------------------------------------------------------------------------------
# user_id;location;age
usuarios = pd.read_csv(PATH+'BX-Users.csv', sep=';')
print(usuarios)

usuarios = usuarios[usuarios['user_id'].isin(ids_usuarios_validos)]  # quedan 6029 usuarios
usuarios = usuarios.reset_index(drop=True)
print(usuarios.shape)  # 6029
print(idx_users_5.shape)  # 6029  todos los usuarios tienen descripción
print("usuarios sin edad:", usuarios.age.isna().sum())  # 1728
moda_edad = usuarios.age.mode()  # la moda es 29
idx_nan = usuarios[usuarios.age.isna()].index
usuarios.loc[idx_nan, 'age'] = moda_edad.values[0]

# ahora voy a sustituir los códigos de los usuarios por los ids del diccionario
us = np.zeros((usuarios.shape[0],), dtype=np.int64)
for i in range(us.shape[0]):
    if i % 100 == 0:
        print(i, dic_usuarios[usuarios.loc[i].user_id])
    us[i] = dic_usuarios[usuarios.loc[i].user_id]
usuarios['user_id'] = us

# 'location' puede tener 3 partes (localidad, stado, país) hay muchos n/a y cosas raras, así que
# me quedo sólo con la localidad del usuario
locations = usuarios['location'].str.split(',', n=1).to_list()
ciudades = []
for loc in locations:
    ciudades.append(loc[0])
usuarios['location'] = ciudades
print(usuarios)
print("Localidades diferentes:", usuarios.location.unique().shape)  # 2617

CATEGORIZAR_LOCALIDADES = False
if CATEGORIZAR_LOCALIDADES:
    # cambio autores por números
    localidades = usuarios.location.unique()
    num_localidades = localidades.shape[0]   # 2617
    dic_localidades = dict(zip(localidades, list(range(num_localidades))))
    # ahora voy a sustituir las localidades de los usuarios por los ids del diccionario
    lo = np.zeros((usuarios.shape[0],), dtype=np.int64)
    for i in range(lo.shape[0]):
        if i % 100 == 0:
            print(i, dic_localidades[usuarios.loc[i].location])
        lo[i] = dic_localidades[usuarios.loc[i].location]
    # y ahora hago un one-hot de autores y lo añado al df
    oh_localidades = to_categorical(lo, num_classes=num_localidades, dtype='int32')
    usuarios[localidades] = oh_localidades

    # y finalmente elimino las columnas no categorizadas
    usuarios.drop(['location'], axis='columns', inplace=True)
    toPickle(PATH+'usuarios_categorizados', usuarios)
else:
    usuarios = getPickle(PATH + 'usuarios_categorizados.pkl')
    print(usuarios)

# usuarios = [6029 rows x 2619 columns] -> [user_id; age; localidades(2617)]


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# MATRIZONAS
CALCULA_MATRIZONAS = False
if CALCULA_MATRIZONAS:
    # Hago la matrizona para codificar los usuarios (con los ratings)
    # hay 6029 usuarios numerados del 1 al 6029
    # hay 5633 items numerados del 1 al 5633
    M = np.zeros((max(valoraciones.user_id), max(valoraciones.item_id)), dtype='int32')
    for i in range(valoraciones.shape[0]):
        user = valoraciones.loc[i, 'user_id']
        item = valoraciones.loc[i, 'item_id']
        M[user-1, item-1] = valoraciones.loc[i, 'rating']
    toPickle('SPARSES/'+PREFIJO+'_MATRIZONA_RATINGS', M)

    # Ahora los convierto a "vio" o no "vio" la película
    M[M > 0] = 1
    toPickle('SPARSES/'+PREFIJO+'_MATRIZONA_VIO_NO_VIO', M)

CREA_CASOS_DE_USO = False
if CREA_CASOS_DE_USO:
    cjto = valoraciones

    # añado las columnas del usuario y del libro con ceros
    col_users = usuarios.columns
    col_users = col_users.drop(['user_id'])
    col_items = libros.columns
    col_items = col_items.drop(['item_id'])

    names = ['user_id', 'item_id', 'rating'] + col_users.to_list() + col_items.to_list()
    ternas = cjto.values
    feat_user = np.zeros((cjto.shape[0], col_users.shape[0]), dtype='int32')
    feat_item = np.zeros((cjto.shape[0], col_items.shape[0]), dtype='int32')
    # se hace más abajo - cjto = pd.DataFrame(np.hstack((ternas, feat_user, feat_item)), columns=names)

    # relleno los usuarios con sus valores
    for i in range(usuarios.shape[0]):
        num_user = usuarios.iloc[i].user_id
        if i % 20 == 0:
            print(i, "/", usuarios.shape[0], "user:", num_user)
        idx = cjto[cjto.user_id == num_user].index
        feat_user[idx, :] = np.tile(usuarios.iloc[i, 1:].to_list(), [idx.shape[0], 1])
        # lento 1 - cjto.loc[cjto.user_id == num_user, col_users] = usuarios.iloc[i, 1:].to_list()
        # lento 2 - cjto.iloc[idx, 3:(3+col_users.shape[0])] = usuarios.iloc[i, 1:].to_list()

    # se almacenan en un fichero los usuarios quitanto el user_id
    usuarios.drop(['user_id'], axis='columns', inplace=True)
    usuarios.to_pickle('SPARSES/'+PREFIJO+'_USUARIOS.pkl')

    # relleno los items con sus valores OJO -> hay 200 libros que no están en 'libros'
    # como feat_item está inicializado a 0, esos 200 libros tendrán todas sus feats a 0
    for i in range(libros.shape[0]):
        num_item = libros.iloc[i].item_id
        if i % 20 == 0:
            print(i, "/", libros.shape[0], "item:", num_item)
        idx = cjto[cjto.item_id == num_item].index
        feat_item[idx, :] = np.tile(libros.iloc[i, 1:].to_list(), [idx.shape[0], 1])
        # lento - cjto.loc[cjto.item_id == num_item, col_items] = libros.iloc[i, 1:].to_list()

    # ahora creo el df con todos los datos
    cjto = pd.DataFrame(np.hstack((ternas, feat_user, feat_item)), columns=names)
    # se almacenas en un fichero los ejemplos
    cjto.to_pickle('SPARSES/'+PREFIJO+'_CASOS_USO.pkl')
else:
    cjto = getPickle('SPARSES/'+PREFIJO+'_CASOS_USO.pkl')


# SEPARAR TRAIN Y TEST
# un porcentaje del total de libros se utiliza sólo para TEST. Esos libros
# no deben aparecer en el TRAIN, así que todos las ternas (user, libro, rating) en
# las que intervienen se van a TEST
porcentaje_test = 0.25
np.random.seed(2032)  # fijo una semilla para que la partición sea siempre la misma
barajado = np.random.permutation((max(cjto.item_id))) + 1  # +1 porque empieza en 0
movies_test = barajado[0:int(max(barajado) * porcentaje_test)]
movies_train = barajado[int(max(barajado) * porcentaje_test):]

# separo las ternas (user, libro, rating)
mask_train = np.in1d(cjto.item_id, movies_train)
mask_test = np.in1d(cjto.item_id, movies_test)
# cjto_train = cjto.iloc[mask_train, 2:]  # quito user_id e item_id
# cjto_test = cjto.iloc[mask_test, 2:]
# cjto_train.to_pickle('SPARSES/'+PREFIJO+'_CASOS_USO_TRAIN.pkl') No los necesito
# cjto_test.to_pickle('SPARSES/'+PREFIJO+'_CASOS_USO_TEST.pkl')
cjto_train = cjto.iloc[mask_train]  # no quito los ids
cjto_test = cjto.iloc[mask_test]
cjto_train.to_pickle('SPARSES/'+PREFIJO+'_CASOS_USO_TRAIN_con_ids.pkl')
cjto_test.to_pickle('SPARSES/'+PREFIJO+'_CASOS_USO_TEST_con_ids.pkl')


# separo las películas
mask_train = np.in1d(libros.item_id, movies_train)
mask_test = np.in1d(libros.item_id, movies_test)
cjto_train = libros.iloc[mask_train, :]
cjto_test = libros.iloc[mask_test, :]
cjto_train.to_pickle('SPARSES/'+PREFIJO+'_BOOKS_TRAIN.pkl')
cjto_test.to_pickle('SPARSES/'+PREFIJO+'_BOOKS_TEST.pkl')


print("FIN")
