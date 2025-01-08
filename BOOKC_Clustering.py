# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
# import sqlite3 as lite
import sys  # time
import pickle  #, cPickle, feather
# import os.path
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import axes3d, Axes3D
import collections
# import random
import fileinput
import h5py

import scipy.stats
# from sklearn import metrics
# from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import MiniBatchKMeans  # , KMeans
# from sklearn.metrics.pairwise import pairwise_distances_argmin
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.decomposition import *

# OJO lo comento porque no soy capaz de instalarlo
# import kmc2

# reload(sys)
# sys.setdefaultencoding('utf8')

PREFIJO = 'BOOKC'

########################################################################################################################
#  Funciones
########################################################################################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printW(x):
    print(bcolors.WARNING + str(x) + bcolors.ENDC)


def printB(x):
    print(bcolors.BOLD + str(x) + bcolors.ENDC)


def printG(x):
    print(bcolors.OKGREEN + str(x) + bcolors.ENDC)


def printE(err):
    print('\033[91m--------------------------------------------\033[0m')
    print('\033[91m' + err + '\033[0m')
    print('\033[91m--------------------------------------------\033[0m')


def toPickle(name, item):
    filehandler = open(name+".pkl", "wb")
    pickle.dump(item, filehandler)
    filehandler.close()


def getPickle(name):
    filehandler = open(name, "rb")
    object = pickle.load(filehandler)
    filehandler.close()

    return object


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def saveTxt(fileName, data):
    np.savetxt(fileName + ".csv", data, delimiter='\t')  # X is an array

    for i, line in enumerate(fileinput.input(fileName + ".csv", inplace=1)):
        sys.stdout.write(line.replace('.', ','))  # replace 'sit' and write


def getMatrix(nclust, KMU, KMA, invert=False, percent=False, fileName=None, plot=False, entropy=False):
    RET = np.zeros((nclust, nclust))

    for i in range(len(KMU)):
        USER_CLUSTER = KMU[i]
        ANUN_CLUSTER = KMA[i]

        RET[USER_CLUSTER, ANUN_CLUSTER] += 1

    if percent:

        for i in range(nclust):

            if invert:
                total = sum(RET[:, i])
                for j in range(nclust):
                    RET[j, i] = RET[j, i] / total
            else:
                total = sum(RET[i, :])
                for j in range(nclust):
                    RET[i, j] = RET[i, j] / total

    if fileName != None:
        saveTxt(fileName, RET)

    if plot:
        fig = plt.figure(figsize=(30, 20))

        ax = fig.add_subplot(111, title=str(nclust)+' CLUSTERS', aspect='equal')
        p = plt.pcolormesh(RET, cmap=cm.get_cmap("OrRd"))
        plt.grid(True)

        plt.gca().invert_yaxis()
        plt.xticks(range(nclust+1))
        plt.yticks(range(nclust+1))

        plt.xlabel("Cluster Película")
        plt.ylabel("Cluster Usuario")

        # plt.show()

        plt.savefig(fileName+'.png',bbox_inches='tight')

    if entropy:
        R1, R2 = getGain(RET)
        print("Ganancia por filas: "+str(R1))
        print("Ganancia por columnas: "+str(R2))

    return RET


def getGain(MT):

    LEN = len(MT)
    RET = np.zeros((LEN, LEN))

    for f in range(LEN):
        total = sum(MT[f, :])

        for c in range(LEN):
            div = MT[f, c]/total

            if div != 0:
                lg = np.log2(div)
            else:
                lg = 0

            res = div*lg
            RET[f, c] = res

    TOT = 0
    countTOT = 0

    for f in range(LEN):
        gainRowSum = -(sum(RET[f, :]))
        countRowSum = sum(MT[f, :])

        countTOT += countRowSum
        TOT += gainRowSum*countRowSum

    R1 = TOT/countTOT

    # Por columnas
    # -------------------------------------------------------------
    LEN = len(MT)
    RET = np.zeros((LEN, LEN))

    for c in range(LEN):
        total = sum(MT[:, c])

        for f in range(LEN):
            div = MT[f, c]/total
            if div != 0:
                lg = np.log2(div)
            else:
                lg = 0
            res = div*lg
            RET[f, c] = res
    TOT = 0
    countTOT = 0

    for c in range(LEN):
        gainColSum = -(sum(RET[:, c]))
        countColSum = sum(MT[:, c])

        countTOT += countColSum
        TOT += gainColSum * countColSum

    R2 = TOT/countTOT

    return R1, R2


def toLat(TEMPLAT):
    return ((TEMPLAT * 2) - 1) * 90


def toLng(TEMPLNG):
    return ((TEMPLNG * 2) - 1) * 180


def getClustersMatrix(XU, XA, doPlot=False, init="k-means++", percent=True, nclust=5, nitmax=10, batch=512, verbose=1,
                      seed=np.random.seed(seed=None)):
    # Compute clustering with MiniBatchKMeans
    # ----------------------------------------
    mbkU = MiniBatchKMeans(init=init, n_clusters=nclust, batch_size=batch, n_init=100, max_no_improvement=nitmax,
                           verbose=verbose, random_state=seed)
    mbkU.fit(XU)

    mbkA = MiniBatchKMeans(init=init, n_clusters=nclust, batch_size=batch, n_init=100, max_no_improvement=nitmax,
                           verbose=verbose,random_state=seed)
    mbkA.fit(XA)

    matName = "ClusteringOutput/KM_"+str(nclust)+"_"

    RET = getMatrix(nclust, mbkU.labels_, mbkA.labels_, plot=doPlot, invert=False, percent=False, fileName=matName+"N",
                    entropy=True)
    RET_P = getMatrix(nclust, mbkU.labels_, mbkA.labels_, plot=doPlot, invert=False, percent=True, fileName=matName+"P")
    RET_P_T = getMatrix(nclust, mbkU.labels_, mbkA.labels_, plot=doPlot, invert=True, percent=True,
                        fileName=matName+"P_T")

    '''
    LATS = [0]*nclust
    LNGS = [0]*nclust

    #Media lat y long por cluster
    for i in range(nclust):
        TEMPLAT = (XU[np.where(mbkU.labels_ == i)[0]][:,0]).toarray()
        TEMPLNG = (XU[np.where(mbkU.labels_ == i)[0]][:,1]).toarray()

        TEMPLAT = ((TEMPLAT * 2) - 1) * 90
        TEMPLNG = ((TEMPLNG * 2) - 1) * 180

        LATS[i] = np.mean(TEMPLAT)
        LNGS[i] = np.mean(TEMPLNG)


        print "-"*50
        print "Cluster "+ str(i)
        print "-"*50
        print "Media"
        print str(np.mean(TEMPLAT[:,0]))+", "+str(np.mean(TEMPLNG[:, 0]))

        comlat = collections.Counter(TEMPLAT[:,0]).most_common(2)
        comlng = collections.Counter(TEMPLNG[:, 0]).most_common(2)

        print "Más común"
        print str(comlat[0][0]) + ", " + str(comlng[0][0])

        print "2º Más común"
        print str(comlat[1][0]) + ", " + str(comlng[1][0])


    print("Usuarios")
    print(collections.Counter(mbkU.labels_))
    print("-")*60
    print("Documentos")
    print(collections.Counter(mbkA.labels_))
    print("-")*60

    np.savetxt('matrix_percent.csv', RET, delimiter=';')  # X is an array
    np.savetxt('matrix_percent_t.csv', RET_T, delimiter=';')  # X is an array
    '''

    return mbkU, mbkA


def doBatchKmeans(XU, clusters=10, btch=512, seed=10):

    kmeans = MiniBatchKMeans(init="k-means++", n_clusters=clusters, random_state=seed, verbose=True)

    btch = 512
    its = XU.shape[0] / btch
    last = XU.shape[0] % btch

    fd = 0
    td = 0

    for i in range(its):
        fd = btch * i
        td = btch * (i + 1)
        kmeans.partial_fit(XU[fd:td])

    kmeans.partial_fit(XU[td:td + last])

    return kmeans


def calcClustNum(XU, clustList=None):

    if clustList == None:
        printE("Es necesario el parámetro clustList")
        exit(1)

    for i in clustList:
        mbkU = MiniBatchKMeans(init='k-means++', n_clusters=i, batch_size=512, n_init=100, max_no_improvement=100,
                               verbose=0)
        mbkU.fit(XU)


def getCountryName(LAT, LNG):
    LTNM = {'32.806671':'Alabama', '61.370716':'Alaska', '33.729759':'Arizona', '34.969704':'Arkansas', '36.116203':'California', '39.059811':'Colorado', '41.597782':'Connecticut', '39.318523':'Delaware', '38.897438':'District of Columbia', '27.766279':'Florida', '33.040619':'Georgia', '21.094318':'Hawaii', '44.240459':'Idaho', '40.349457':'Illinois', '39.849426':'Indiana', '42.011539':'Iowa', '38.5266':'Kansas', '37.66814':'Kentucky', '31.169546':'Louisiana', '44.693947':'Maine', '39.063946':'Maryland', '42.230171':'Massachusetts', '43.326618':'Michigan', '45.694454':'Minnesota', '32.741646':'Mississippi', '38.456085':'Missouri', '46.921925':'Montana', '41.12537':'Nebraska', '38.313515':'Nevada', '43.452492':'New Hampshire', '40.298904':'New Jersey', '34.840515':'New Mexico', '42.165726':'New York', '35.630066':'North Carolina', '47.528912':'North Dakota', '40.388783':'Ohio', '35.565342':'Oklahoma', '44.572021':'Oregon', '40.590752':'Pennsylvania', '41.680893':'Rhode Island', '33.856892':'South Carolina', '44.299782':'South Dakota', '35.747845':'Tennessee', '31.054487':'Texas', '40.150032':'Utah', '44.045876':'Vermont', '37.769337':'Virginia', '47.400902':'Washington', '38.491226':'West Virginia', '44.268543':'Wisconsin', '42.755966':'Wyoming', '39.828175':'Armed Forces Americas', '39.828175':'Armed Forces Europe', '39.828175':'Armed Forces Pacific', '33.93911':' Afghanistan', '37.0625':' Åland Islands', '41.153332':' Albania', '28.033886':' Algeria', '-14.270972':' American Samoa', '42.546245':' Andorra', '-11.202692':' Angola', '18.220554':' Anguilla', '-75.250973':' Antarctica', '17.060816':' Antigua and Barbuda', '-38.416097':' Argentina', '40.069099':' Armenia', '12.52111':' Aruba', '-25.274398':' Australia', '47.516231':' Austria', '40.143105':' Azerbaijan', '25.03428':' Bahamas', '25.930414':' Bahrain', '23.684994':' Bangladesh', '13.193887':' Barbados', '53.709807':' Belarus', '50.503887':' Belgium', '17.189877':' Belize', '9.30769':' Benin', '32.321384':' Bermuda', '27.514162':' Bhutan', '-16.290154':' Bolivia, Plurinational State of', '43.915886':' Bosnia and Herzegovina', '-22.328474':' Botswana', '-54.423199':' Bouvet Island', '-14.235004':' Brazil', '-6.343194':' British Indian Ocean Territory', '4.535277':' Brunei Darussalam', '42.733883':' Bulgaria', '12.238333':' Burkina Faso', '-3.373056':' Burundi', '12.565679':' Cambodia', '7.369722':' Cameroon', '56.130366':' Canada', '16.002082':' Cape Verde', '19.513469':' Cayman Islands', '6.611111':' Central African Republic', '15.454166':' Chad', '-35.675147':' Chile', '35.86166':' China', '-10.447525':' Christmas Island', '37.0625':' Cocos (Keeling) Islands', '4.570868':' Colombia', '-11.875001':' Comoros', '-0.228021':' Congo', '-0.228021':' Congo, the Democratic Republic of the', '-21.236736':' Cook Islands', '9.748917':' Costa Rica', '7.539989':' Côte d\'Ivoire', '45.1':' Croatia', '21.521757':' Cuba', '35.126413':' Cyprus', '49.817492':' Czech Republic', '56.26392':' Denmark', '11.825138':' Djibouti', '15.414999':' Dominica', '18.735693':' Dominican Republic', '-1.831239':' Ecuador', '26.820553':' Egypt', '13.794185':' El Salvador', '1.650801':' Equatorial Guinea', '15.179384':' Eritrea', '58.595272':' Estonia', '9.145':' Ethiopia', '-51.796253':' Falkland Islands (Malvinas)', '61.892635':' Faroe Islands', '-16.578193':' Fiji', '61.92411':' Finland', '46.227638':' France', '3.933889':' French Guiana', '-17.679742':' French Polynesia', '37.0625':' French Southern Territories', '-0.803689':' Gabon', '13.443182':' Gambia', '32.157435':' Georgia', '51.165691':' Germany', '7.946527':' Ghana', '36.137741':' Gibraltar', '39.074208':' Greece', '71.706936':' Greenland', '12.262776':' Grenada', '16.995971':' Guadeloupe', '13.444304':' Guam', '15.783471':' Guatemala', '49.465691':' Guernsey', '9.945587':' Guinea', '11.803749':' Guinea-Bissau', '4.860416':' Guyana', '18.971187':' Haiti', '-53.08181':' Heard Island and McDonald Islands', '37.0625':' Holy See (Vatican City State)', '15.199999':' Honduras', '22.396428':' Hong Kong', '47.162494':' Hungary', '64.963051':' Iceland', '20.593684':' India', '-0.789275':' Indonesia', '32.427908':' Iran, Islamic Republic of', '33.223191':' Iraq', '53.41291':' Ireland', '54.236107':' Isle of Man', '31.046051':' Israel', '41.87194':' Italy', '18.109581':' Jamaica', '36.204824':' Japan', '49.214439':' Jersey', '30.585164':' Jordan', '48.019573':' Kazakhstan', '-0.023559':' Kenya', '-3.370417':' Kiribati', '35.907757':' Korea, Democratic People\'s Republic of', '35.907757':' Korea, Republic of', '29.31166':' Kuwait', '41.20438':' Kyrgyzstan', '19.85627':' Lao People\'s Democratic Republic', '56.879635':' Latvia', '33.854721':' Lebanon', '-29.609988':' Lesotho', '6.428055':' Liberia', '37.0625':' Libyan Arab Jamahiriya', '47.166':' Liechtenstein', '55.169438':' Lithuania', '49.815273':' Luxembourg', '22.198745':' Macao', '41.608635':' Macedonia, the former Yugoslav Republic of', '-18.766947':' Madagascar', '-13.254308':' Malawi', '4.210484':' Malaysia', '3.202778':' Maldives', '17.570692':' Mali', '35.937496':' Malta', '7.131474':' Marshall Islands', '14.641528':' Martinique', '21.00789':' Mauritania', '-20.348404':' Mauritius', '-12.8275':' Mayotte', '23.634501':' Mexico', '7.425554':' Micronesia, Federated States of', '47.411631':' Moldova, Republic of', '43.750298':' Monaco', '46.862496':' Mongolia', '42.708678':' Montenegro', '16.742498':' Montserrat', '31.791702':' Morocco', '-18.665695':' Mozambique', '21.913965':' Myanmar', '-22.95764':' Namibia', '-0.522778':' Nauru', '28.394857':' Nepal', '52.132633':' Netherlands', '12.226079':' Netherlands Antilles', '-20.904305':' New Caledonia', '-40.900557':' New Zealand', '12.865416':' Nicaragua', '17.607789':' Niger', '9.081999':' Nigeria', '-19.054445':' Niue', '-29.040835':' Norfolk Island', '17.33083':' Northern Mariana Islands', '60.472024':' Norway', '21.512583':' Oman', '30.375321':' Pakistan', '7.51498':' Palau', '42.094445':' Palestinian Territory, Occupied', '8.537981':' Panama', '-6.314993':' Papua New Guinea', '-23.442503':' Paraguay', '-9.189967':' Peru', '12.879721':' Philippines', '-24.703615':' Pitcairn', '51.919438':' Poland', '39.399872':' Portugal', '18.220833':' Puerto Rico', '25.354826':' Qatar', '-21.115141':' Réunion', '45.943161':' Romania', '61.52401':' Russian Federation', '-1.940278':' Rwanda', '37.0625':' Saint Barthélemy', '-24.143474':' Saint Helena, Ascension and Tristan da Cunha', '17.357822':' Saint Kitts and Nevis', '13.909444':' Saint Lucia', '43.589046':' Saint Martin (French part)', '46.941936':' Saint Pierre and Miquelon', '12.984305':' Saint Vincent and the Grenadines', '-13.759029':' Samoa', '43.94236':' San Marino', '0.18636':' Sao Tome and Principe', '23.885942':' Saudi Arabia', '14.497401':' Senegal', '44.016521':' Serbia', '-4.679574':' Seychelles', '8.460555':' Sierra Leone', '1.352083':' Singapore', '48.669026':' Slovakia', '46.151241':' Slovenia', '-9.64571':' Solomon Islands', '5.152149':' Somalia', '-30.559482':' South Africa', '-54.429579':' South Georgia and the South Sandwich Islands', '40.463667':' Spain', '7.873054':' Sri Lanka', '12.862807':' Sudan', '3.919305':' Suriname', '77.553604':' Svalbard and Jan Mayen', '-26.522503':' Swaziland', '60.128161':' Sweden', '46.818188':' Switzerland', '34.802075':' Syrian Arab Republic', '23.69781':' Taiwan, Province of China', '38.861034':' Tajikistan', '-6.369028':' Tanzania, United Republic of', '15.870032':' Thailand', '-8.874217':' Timor-Leste', '8.619543':' Togo', '-8.967363':' Tokelau', '-21.178986':' Tonga', '10.691803':' Trinidad and Tobago', '33.886917':' Tunisia', '38.963745':' Turkey', '38.969719':' Turkmenistan', '21.694025':' Turks and Caicos Islands', '-7.109535':' Tuvalu', '1.373333':' Uganda', '48.379433':' Ukraine', '23.424076':' United Arab Emirates', '55.378051':' United Kingdom', '37.09024':' United States', '24.747346':' United States Minor Outlying Islands', '-32.522779':' Uruguay', '41.377491':' Uzbekistan', '-15.376706':' Vanuatu', '6.42375':' Venezuela, Bolivarian Republic of', '14.058324':' Viet Nam', '18.335765':' Virgin Islands, British', '18.335765':' Virgin Islands, U.S.', '-13.768752':' Wallis and Futuna', '24.215527':' Western Sahara', '15.552727':' Yemen', '-13.133897':' Zambia', '-19.015438':'Zimbabwe', '50.848307':'European Union', '46.227638':'Metropolitan France', '18.736462':'Asia/Pacific Region', '0':'Anonymous Proxy'}
    LGNM = {'-86.79113':'Alabama', '-152.404419':'Alaska', '-111.431221':'Arizona', '-92.373123':'Arkansas', '-119.681564':'California', '-105.311104':'Colorado', '-72.755371':'Connecticut', '-75.507141':'Delaware', '-77.026817':'District of Columbia', '-81.686783':'Florida', '-83.643074':'Georgia', '-157.498337':'Hawaii', '-114.478828':'Idaho', '-88.986137':'Illinois', '-86.258278':'Indiana', '-93.210526':'Iowa', '-96.726486':'Kansas', '-84.670067':'Kentucky', '-91.867805':'Louisiana', '-69.381927':'Maine', '-76.802101':'Maryland', '-71.530106':'Massachusetts', '-84.536095':'Michigan', '-93.900192':'Minnesota', '-89.678696':'Mississippi', '-92.288368':'Missouri', '-110.454353':'Montana', '-98.268082':'Nebraska', '-117.055374':'Nevada', '-71.563896':'New Hampshire', '-74.521011':'New Jersey', '-106.248482':'New Mexico', '-74.948051':'New York', '-79.806419':'North Carolina', '-99.784012':'North Dakota', '-82.764915':'Ohio', '-96.928917':'Oklahoma', '-122.070938':'Oregon', '-77.209755':'Pennsylvania', '-71.51178':'Rhode Island', '-80.945007':'South Carolina', '-99.438828':'South Dakota', '-86.692345':'Tennessee', '-97.563461':'Texas', '-111.862434':'Utah', '-72.710686':'Vermont', '-78.169968':'Virginia', '-121.490494':'Washington', '-80.954453':'West Virginia', '-89.616508':'Wisconsin', '-107.30249':'Wyoming', '-98.5795':'Armed Forces Americas', '-98.5795':'Armed Forces Europe', '-98.5795':'Armed Forces Pacific', '67.709953':' Afghanistan', '-95.677068':' Åland Islands', '20.168331':' Albania', '1.659626':' Algeria', '-170.132217':' American Samoa', '1.601554':' Andorra', '17.873887':' Angola', '-63.068615':' Anguilla', '-0.071389':' Antarctica', '-61.796428':' Antigua and Barbuda', '-63.616672':' Argentina', '45.038189':' Armenia', '-69.968338':' Aruba', '133.775136':' Australia', '14.550072':' Austria', '47.576927':' Azerbaijan', '-77.39628':' Bahamas', '50.637772':' Bahrain', '90.356331':' Bangladesh', '-59.543198':' Barbados', '27.953389':' Belarus', '4.469936':' Belgium', '-88.49765':' Belize', '2.315834':' Benin', '-64.75737':' Bermuda', '90.433601':' Bhutan', '-63.588653':' Bolivia, Plurinational State of', '17.679076':' Bosnia and Herzegovina', '24.684866':' Botswana', '3.413194':' Bouvet Island', '-51.92528':' Brazil', '71.876519':' British Indian Ocean Territory', '114.727669':' Brunei Darussalam', '25.48583':' Bulgaria', '-1.561593':' Burkina Faso', '29.918886':' Burundi', '104.990963':' Cambodia', '12.354722':' Cameroon', '-106.346771':' Canada', '-24.013197':' Cape Verde', '-80.566956':' Cayman Islands', '20.939444':' Central African Republic', '18.732207':' Chad', '-71.542969':' Chile', '104.195397':' China', '105.690449':' Christmas Island', '-95.677068':' Cocos (Keeling) Islands', '-74.297333':' Colombia', '43.872219':' Comoros', '15.827659':' Congo', '15.827659':' Congo, the Democratic Republic of the', '-159.777671':' Cook Islands', '-83.753428':' Costa Rica', '-5.54708':' Côte d\'Ivoire', '15.2':' Croatia', '-77.781167':' Cuba', '33.429859':' Cyprus', '15.472962':' Czech Republic', '9.501785':' Denmark', '42.590275':' Djibouti', '-61.370976':' Dominica', '-70.162651':' Dominican Republic', '-78.183406':' Ecuador', '30.802498':' Egypt', '-88.89653':' El Salvador', '10.267895':' Equatorial Guinea', '39.782334':' Eritrea', '25.013607':' Estonia', '40.489673':' Ethiopia', '-59.523613':' Falkland Islands (Malvinas)', '-6.911806':' Faroe Islands', '179.414413':' Fiji', '25.748151':' Finland', '2.213749':' France', '-53.125782':' French Guiana', '-149.406843':' French Polynesia', '-95.677068':' French Southern Territories', '11.609444':' Gabon', '-15.310139':' Gambia', '-82.907123':' Georgia', '10.451526':' Germany', '-1.023194':' Ghana', '-5.345374':' Gibraltar', '21.824312':' Greece', '-42.604303':' Greenland', '-61.604171':' Grenada', '-62.067641':' Guadeloupe', '144.793731':' Guam', '-90.230759':' Guatemala', '-2.585278':' Guernsey', '-9.696645':' Guinea', '-15.180413':' Guinea-Bissau', '-58.93018':' Guyana', '-72.285215':' Haiti', '73.504158':' Heard Island and McDonald Islands', '-95.677068':' Holy See (Vatican City State)', '-86.241905':' Honduras', '114.109497':' Hong Kong', '19.503304':' Hungary', '-19.020835':' Iceland', '78.96288':' India', '113.921327':' Indonesia', '53.688046':' Iran, Islamic Republic of', '43.679291':' Iraq', '-8.24389':' Ireland', '-4.548056':' Isle of Man', '34.851612':' Israel', '12.56738':' Italy', '-77.297508':' Jamaica', '138.252924':' Japan', '-2.13125':' Jersey', '36.238414':' Jordan', '66.923684':' Kazakhstan', '37.906193':' Kenya', '-168.734039':' Kiribati', '127.766922':' Korea, Democratic People\'s Republic of', '127.766922':' Korea, Republic of', '47.481766':' Kuwait', '74.766098':' Kyrgyzstan', '102.495496':' Lao People\'s Democratic Republic', '24.603189':' Latvia', '35.862285':' Lebanon', '28.233608':' Lesotho', '-9.429499':' Liberia', '-95.677068':' Libyan Arab Jamahiriya', '9.555373':' Liechtenstein', '23.881275':' Lithuania', '6.129583':' Luxembourg', '113.543873':' Macao', '21.745275':' Macedonia, the former Yugoslav Republic of', '46.869107':' Madagascar', '34.301525':' Malawi', '101.975766':' Malaysia', '73.22068':' Maldives', '-3.996166':' Mali', '14.375416':' Malta', '171.184478':' Marshall Islands', '-61.024174':' Martinique', '-10.940835':' Mauritania', '57.552152':' Mauritius', '45.166244':' Mayotte', '-102.552784':' Mexico', '150.550812':' Micronesia, Federated States of', '28.369885':' Moldova, Republic of', '7.412841':' Monaco', '103.846656':' Mongolia', '19.37439':' Montenegro', '-62.187366':' Montserrat', '-7.09262':' Morocco', '35.529562':' Mozambique', '95.956223':' Myanmar', '18.49041':' Namibia', '166.931503':' Nauru', '84.124008':' Nepal', '5.291266':' Netherlands', '-69.060087':' Netherlands Antilles', '165.618042':' New Caledonia', '174.885971':' New Zealand', '-85.207229':' Nicaragua', '8.081666':' Niger', '8.675277':' Nigeria', '-169.867233':' Niue', '167.954712':' Norfolk Island', '145.38469':' Northern Mariana Islands', '8.468946':' Norway', '55.923255':' Oman', '69.345116':' Pakistan', '134.58252':' Palau', '17.266614':' Palestinian Territory, Occupied', '-80.782127':' Panama', '143.95555':' Papua New Guinea', '-58.443832':' Paraguay', '-75.015152':' Peru', '121.774017':' Philippines', '-127.439308':' Pitcairn', '19.145136':' Poland', '-8.224454':' Portugal', '-66.590149':' Puerto Rico', '51.183884':' Qatar', '55.536384':' Réunion', '24.96676':' Romania', '105.318756':' Russian Federation', '29.873888':' Rwanda', '-95.677068':' Saint Barthélemy', '-10.030696':' Saint Helena, Ascension and Tristan da Cunha', '-62.782998':' Saint Kitts and Nevis', '-60.978893':' Saint Lucia', '5.885031':' Saint Martin (French part)', '-56.27111':' Saint Pierre and Miquelon', '-61.287228':' Saint Vincent and the Grenadines', '-172.104629':' Samoa', '12.457777':' San Marino', '6.613081':' Sao Tome and Principe', '45.079162':' Saudi Arabia', '-14.452362':' Senegal', '21.005859':' Serbia', '55.491977':' Seychelles', '-11.779889':' Sierra Leone', '103.819836':' Singapore', '19.699024':' Slovakia', '14.995463':' Slovenia', '160.156194':' Solomon Islands', '46.199616':' Somalia', '22.937506':' South Africa', '-36.587909':' South Georgia and the South Sandwich Islands', '-3.74922':' Spain', '80.771797':' Sri Lanka', '30.217636':' Sudan', '-56.027783':' Suriname', '23.670272':' Svalbard and Jan Mayen', '31.465866':' Swaziland', '18.643501':' Sweden', '8.227512':' Switzerland', '38.996815':' Syrian Arab Republic', '120.960515':' Taiwan, Province of China', '71.276093':' Tajikistan', '34.888822':' Tanzania, United Republic of', '100.992541':' Thailand', '125.727539':' Timor-Leste', '0.824782':' Togo', '-171.855881':' Tokelau', '-175.198242':' Tonga', '-61.222503':' Trinidad and Tobago', '9.537499':' Tunisia', '35.243322':' Turkey', '59.556278':' Turkmenistan', '-71.797928':' Turks and Caicos Islands', '177.64933':' Tuvalu', '32.290275':' Uganda', '31.16558':' Ukraine', '53.847818':' United Arab Emirates', '-3.435973':' United Kingdom', '-95.712891':' United States', '-167.594906':' United States Minor Outlying Islands', '-55.765835':' Uruguay', '64.585262':' Uzbekistan', '166.959158':' Vanuatu', '-66.58973':' Venezuela, Bolivarian Republic of', '108.277199':' Viet Nam', '-64.896335':' Virgin Islands, British', '-64.896335':' Virgin Islands, U.S.', '-177.156097':' Wallis and Futuna', '-12.885834':' Western Sahara', '48.516388':' Yemen', '27.849332':' Zambia', '29.154857':'Zimbabwe', '4.351755':'European Union', '2.213749':'Metropolitan France', '135.476132':'Asia/Pacific Region', '0':'Anonymous Proxy',}

    if LTNM.get(LAT) == LGNM.get(LNG):

        if LTNM.get(LAT) != None :
            return LTNM.get(LAT).strip()

    return "Unknown"


def getUsersClusterInfo(XU, KMU, filename="ClusterInfo"):
    nclust = max(KMU)+1

    file = open(filename+".CSV", "w")

    file.write("CLUSTER;LOCATION;LAT;LNG;TIMES;STDLAT;STDLNG\n")

    for c in range(nclust):

        TEMPLAT = (XU[np.where(KMU == c)[0]][:, 0]).toarray()
        TEMPLNG = (XU[np.where(KMU == c)[0]][:, 1]).toarray()

        STDLAT = str(np.std(toLat(TEMPLAT)))
        STDLNG = str(np.std(toLng(TEMPLNG)))

        # mcomm = len(collections.Counter(TEMPLAT[:,0]).most_common())
        mcomm = 1

        comlat = collections.Counter(TEMPLAT[:, 0]).most_common()
        comlng = collections.Counter(TEMPLNG[:, 0]).most_common()

        for mc in range(mcomm):
            LAT = str(toLat(comlat[mc][0]))
            LNG = str(toLng(comlng[mc][0]))
            times = comlat[mc][1]

            line = str(c)+";"+getCountryName(LAT, LNG)+";"+LAT+";"+LNG+";"+str(times)+";"+STDLAT+";"+STDLNG+"\n"
            file.write(line)

    file.close()

    for i, line in enumerate(fileinput.input(filename + ".csv", inplace=1)):
        sys.stdout.write(line.replace('.', ','))  # replace 'sit' and write

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def getIncertidumbre(Kusers, Kitems, VALS, USRLABS, ADVLABS, fileName=None):

    P = np.zeros((Kusers, Kitems))
    INC = np.zeros((Kusers, Kitems))

    ROWINCS = [0] * Kusers

    for i in range(Kusers):
        # Se obtienen las valoraciones de los usuarios del cluster i para todos los centroides de anuncios
        VALCLUST = VALS[np.where(USRLABS == i)]

        ROWINCSUM = 0
        ROWTOTSUM = 0
        ITEMS_CLUSTER_USRS = VALCLUST.shape[0]

        for j in range(Kitems):
            # Para cada uno de los clusters de anuncios (j) se suman los mayores o iguales a 0
            COUNTS = (VALCLUST[:, j] >= 0).sum()
            # Tamaño del cluster de adv j
            SIZE_CLUST_ADV = len(np.where(ADVLABS == j)[0])
            # Proporción de items positivos: El valor de la celda serán el numero de items mayores de 0 partido del
            # número total de items de la celda
            CELL = COUNTS * 1.0 / ITEMS_CLUSTER_USRS
            P[i, j] = CELL

            # Se calcula la incertidumbre de la celda => I(i,j) = -p(i,j)*log2(p(i,j)) - (1-p(i,j))*log2(1-p(i,j))
            if CELL == 1:
                INC[i, j] = ((-CELL) * np.log2(CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV
            elif CELL == 0:
                INC[i, j] = ((1 - CELL) * np.log2(1 - CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV
            else:
                INC[i, j] = ((-CELL) * np.log2(CELL)) - ((1 - CELL) * np.log2(1 - CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV

        ROWINCS[i] = (ROWINCSUM / ROWTOTSUM) * ITEMS_CLUSTER_USRS

    print("\t· Incertidumbre: " + str(sum(ROWINCS) / len(USRLABS)))

    if fileName != None:
        saveTxt(fileName, INC)
        toPickle(fileName, INC)

    return INC

'''def getIncertidumbre(K, VALS, USRLABS, ADVLABS, fileName=None):

    P = np.zeros((K, K))
    INC = np.zeros((K, K))

    ROWINCS = [0] * K

    for i in range(K):
        # Se obtienen las valoraciones de los usuarios del cluster i para todos los centroides de anuncios
        VALCLUST = VALS[np.where(USRLABS == i)]

        ROWINCSUM = 0
        ROWTOTSUM = 0
        ITEMS_CLUSTER_USRS = VALCLUST.shape[0]

        for j in range(K):
            # Para cada uno de los clusters de anuncios (j) se suman los mayores o iguales a 0
            COUNTS = (VALCLUST[:, j] >= 0).sum()
            # Tamaño del cluster de adv j
            SIZE_CLUST_ADV = len(np.where(ADVLABS == j)[0])
            # Proporción de items positivos: El valor de la celda serán el numero de items mayores de 0 partido del
            # número total de items de la celda
            CELL = COUNTS * 1.0 / ITEMS_CLUSTER_USRS
            P[i, j] = CELL

            # Se calcula la incertidumbre de la celda => I(i,j) = -p(i,j)*log2(p(i,j)) - (1-p(i,j))*log2(1-p(i,j))
            if CELL == 1:
                INC[i, j] = ((-CELL) * np.log2(CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV
            elif CELL == 0:
                INC[i, j] = ((1 - CELL) * np.log2(1 - CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV
            else:
                INC[i, j] = ((-CELL) * np.log2(CELL)) - ((1 - CELL) * np.log2(1 - CELL))
                ROWINCSUM += INC[i, j] * SIZE_CLUST_ADV
                ROWTOTSUM += SIZE_CLUST_ADV

        ROWINCS[i] = (ROWINCSUM / ROWTOTSUM) * ITEMS_CLUSTER_USRS

    print("\t· Incertidumbre: " + str(sum(ROWINCS) / len(USRLABS)))

    if fileName != None:
        saveTxt(fileName, INC)
        toPickle(fileName, INC)

    return INC
'''

def getMethodData():

    # Cargar los datos

    # películas del test RAW
    ADS_RAW = getPickle('SPARSES/'+PREFIJO+'_BOOKS_TEST.pkl')

    # usuarios RAW
    USR_RAW = getPickle('SPARSES/'+PREFIJO+'_USUARIOS.pkl')

    # valoraciones que dan los usuarios a los centroides de las películas en TRAIN
    h5f = h5py.File('VALORACIONES/'+PREFIJO+'_VAL_POR_CENTROIDE_RAW_TRAIN.h5', 'r')
    VAL_TRAIN = h5f['VAL_POR_CENTROIDE'][:]
    h5f.close()

    # valoraciones que dan los usuarios a los centroides de las películas en TEST
    h5f = h5py.File('VALORACIONES/'+PREFIJO+'_VAL_POR_CENTROIDE_RAW_TEST.h5', 'r')
    VAL_TEST = h5f['VAL_POR_CENTROIDE'][:]
    h5f.close()

    return USR_RAW, ADS_RAW,  VAL_TRAIN, VAL_TEST


def performTestT(IM1, IM2, verbose=False):

    IM1 = IM1.reshape((1, Kusers * Kusers))[0]
    IM2 = IM2.reshape((1, Kusers * Kusers))[0]

    TESTT = scipy.stats.ttest_rel(IM1, IM2)

    if verbose:
        print("\t"+str(TESTT[1] / 2))

    return TESTT[1] / 2


def performAllTests(day=1):

    def fmt(d):
        return str(d).replace(".", ",")

    DAY = "DAY"+str(day)
    DIA = "DIA "+str(day)

    # print(DIA+", MÉTODO 2 RAW VS MÉTODO 3 RAW")
    IM1 = getPickle("ClusteringResults/Inc/METHOD2_"+DAY+"_RAW_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD3_"+DAY+"_RAW_INC")
    R1 = performTestT(IM1, IM2)

    # print(DIA+", MÉTODO 2 RAW VS MÉTODO 4")
    IM1 = getPickle("ClusteringResults/Inc/METHOD2_"+DAY+"_RAW_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD4_"+DAY+"_VAL_INC")
    R2 = performTestT(IM1, IM2)

    # print(DIA+", MÉTODO 2 RAW VS MÉTODO 4.2")
    IM1 = getPickle("ClusteringResults/Inc/METHOD2_"+DAY+"_RAW_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD4.2_"+DAY+"_VALPOND_INC")
    R3 = performTestT(IM1, IM2)

    # print(DIA+", MÉTODO 3 RAW VS MÉTODO 4")
    IM1 = getPickle("ClusteringResults/Inc/METHOD3_"+DAY+"_RAW_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD4_"+DAY+"_VAL_INC")
    R4 = performTestT(IM1, IM2)

    # print(DIA+", MÉTODO 3 RAW VS MÉTODO 4,2")
    IM1 = getPickle("ClusteringResults/Inc/METHOD3_"+DAY+"_RAW_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD4.2_"+DAY+"_VALPOND_INC")
    R5 = performTestT(IM1, IM2)

    # print(DIA+", MÉTODO 4 RAW VS MÉTODO 4,2")
    IM1 = getPickle("ClusteringResults/Inc/METHOD4_"+DAY+"_VAL_INC")
    IM2 = getPickle("ClusteringResults/Inc/METHOD4.2_"+DAY+"_VALPOND_INC")
    R6 = performTestT(IM1, IM2)

    print(DIA)
    print("-"*45)
    print("-\t"+fmt(R1)+"\t"+fmt(R2)+"\t"+fmt(R3))
    print(fmt(R1)+"\t-\t"+fmt(R4)+"\t"+fmt(R5))
    print(fmt(R2)+"\t"+fmt(R4)+"\t-\t"+fmt(R6))
    print(fmt(R3)+"\t"+fmt(R5)+"\t"+fmt(R6)+"\t-")


def getDIFFcentroids(K=100, SEED=100, verbose=1, testData=False):
    # Calcula centroides de películas

    # Usar train o test para hacer clustering
    if testData:
        FILENAME = "DIFFS/"+PREFIJO+"_BOOKS_RAW_CENTROIDS_TEST"
        cjto = getPickle('SPARSES/'+PREFIJO+'_BOOKS_TEST.pkl')
    else:
        FILENAME = "DIFFS/"+PREFIJO+"_BOOKS_RAW_CENTROIDS_TRAIN"
        cjto = getPickle('SPARSES/'+PREFIJO+'_BOOKS_TRAIN.pkl')

    # no necesito el item_id
    cjto.drop(['item_id'], axis='columns', inplace=True)

    # Clustering de películas
    model_movies_RAW = MiniBatchKMeans(K, init='k-means++', max_no_improvement=1000, batch_size=1024, verbose=verbose,
                                       random_state=SEED, max_iter=1000)
    model_movies_RAW.fit(cjto)

    toPickle(FILENAME, model_movies_RAW.cluster_centers_)
    toPickle(FILENAME.replace("CENTROIDS", "LABELS"), model_movies_RAW.labels_)


def methodTwo(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):
    # Cluster con los usuarios en crudo
    # En el artículo: Cl(Uraw)
    # Crea clusters si no existen y calcula la incertidumbre del método

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    '''# estandarizo las columnas que tienen valores muy grandes
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    col_a_estandarizar = ['age', 'latitude', 'longitude']
    features = USR_RAW[col_a_estandarizar]
    scaler.fit(features)  # entreno sobre el train
    features = scaler.transform(features)
    USR_RAW[col_a_estandarizar] = features
    '''

    print("Datos cargados...")

    #####################################################################

    if not loadData:
        print("Centroides generados...")
        modelUSR_RAW = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelUSR_RAW.fit(USR_RAW)

        model_USR_RAW_labels = modelUSR_RAW.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_"+str(Kusers)+"_USR_RAW_labels", model_USR_RAW_labels)

    else:

        model_USR_RAW_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_"+str(Kusers)+"_USR_RAW_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Usuarios RAW , Películas RAW")
    print("U_raw")
    IM1 = getIncertidumbre(Kusers, Kitems, VAL_TEST, model_USR_RAW_labels, model_DIFF_RAW_labels,
                           fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD2_RAW_INC")
    print("-" * 65)


def methodThree(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):
    # Cluster con los usuarios en crudo + Valoración de cada cluster
    # En el artículo: Cl(Uraw + Uti)
    # Crea clusters si no existen y calcula la incertidumbre del método

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    USR = np.zeros((USR_RAW.shape[0], USR_RAW.shape[1]+VAL_TRAIN.shape[1]))

    '''# estandarizo las columnas que tienen valores muy grandes
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    col_a_estandarizar = ['age', 'latitude', 'longitude']
    features = USR_RAW[col_a_estandarizar]
    scaler.fit(features)  # entreno sobre el train
    features = scaler.transform(features)
    USR_RAW[col_a_estandarizar] = features
    '''

    USR[:, 0:USR_RAW.shape[1]] = USR_RAW
    USR[:, USR_RAW.shape[1]:] = VAL_TRAIN


    print("Datos cargados...")

    #####################################################################

    if not loadData:
        print("Centroides generados...")

        modelUSR_RAW_VAL = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                           verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelUSR_RAW_VAL.fit(USR)

        model_USR_RAW_VAL_labels = modelUSR_RAW_VAL.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_"+str(Kusers)+"_USR_RAW_VAL_labels", model_USR_RAW_VAL_labels)

    else:
        model_USR_RAW_VAL_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_"+str(Kusers)+"_USR_RAW_VAL_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Usuarios RAW + valoración RAW (%), Películas RAW")
    print("U_raw_uti")
    getIncertidumbre(Kusers, Kitems, VAL_TEST, model_USR_RAW_VAL_labels, model_DIFF_RAW_labels,
                     fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD3_RAW_INC")
    print("-" * 65)


def methodFour(Kusers=100, Kitems=100,  SEED=100, verbose=1, loadData=False, max_iter=100):

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    print("Datos cargados...")

    #####################################################################

    if not loadData:

        print("Centroides generados...")
        modelVAL_RAW = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelVAL_RAW.fit(VAL_TRAIN)

        model_VAL_RAW_labels = modelVAL_RAW.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_VAL_RAW_labels", model_VAL_RAW_labels)

    else:
        model_VAL_RAW_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_VAL_RAW_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Valoración RAW (%), Películas RAW")
    print("U_uti")
    getIncertidumbre(Kusers, Kitems, VAL_TEST, model_VAL_RAW_labels, model_DIFF_RAW_labels,
                     fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD4_VAL_INC")
    print("=" * 65)


def methodFourDotTwo(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    CLUSTER_SIZE = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TRAIN.pkl")
    # CLUSTER_SIZE = dict(collections.Counter(CLUSTER_SIZE).most_common(K)).values()
    num_cluster, num_elem_cluster = np.unique(CLUSTER_SIZE, return_counts=True)
    num_elem_total = sum(num_elem_cluster)

    # Se multiplica cada columna por la raiz cuadrada del tamaño del cluster
    # for c in range(K):
    for c in num_cluster:
        # VAL_TRAIN[:, c] = VAL_TRAIN[:, c]*np.sqrt(CLUSTER_SIZE[c]/(sum(CLUSTER_SIZE)*1.0))
        VAL_TRAIN[:, c] = VAL_TRAIN[:, c] * np.sqrt(num_elem_cluster[c] / num_elem_total)

    print("Datos cargados...")

    #####################################################################

    if not loadData:

        print("Centroides generados...")

        modelVAL_RAW = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelVAL_RAW.fit(VAL_TRAIN)

        model_VAL_RAW_labels = modelVAL_RAW.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_VALPOND_RAW_labels", model_VAL_RAW_labels)

    else:
        model_VAL_RAW_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_VALPOND_RAW_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Valoración RAW (%), Películas RAW")
    print("U_w_uti")
    getIncertidumbre(Kusers, Kitems, VAL_TEST, model_VAL_RAW_labels, model_DIFF_RAW_labels,
                     fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD4.2_VALPOND_INC")
    print("=" * 65)


def methodEMB(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):
    # Cluster con los embeddings
    # En el artículo si se pone: Cl(Uemb)
    # Crea clusters si no existen y calcula la incertidumbre del método

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    h5f = h5py.File('EMBEDDINGS/'+PREFIJO+'_EMBEDDING_USUARIOS.h5', 'r')
    USR_EMB = h5f['embedding'][:]
    h5f.close()

    print("Datos cargados...")

    #####################################################################

    if not loadData:
        print("Centroides generados...")

        modelUSR_EMB = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelUSR_EMB.fit(USR_EMB)

        model_USR_EMB_labels = modelUSR_EMB.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_EMB_labels", model_USR_EMB_labels)

    else:

        model_USR_EMB_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_EMB_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Usuarios EMB , Películas RAW")
    print("U_emb")
    IM1 = getIncertidumbre(Kusers, Kitems, VAL_TEST, model_USR_EMB_labels, model_DIFF_RAW_labels,
                           fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD_EMB_RAW_INC")
    print("-" * 65)


def methodRatings(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):
    # Cluster con la matrizona (ratings)
    # En el artículo si se pone:
    # Crea clusters si no existen y calcula la incertidumbre del método

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    USR_RAT = getPickle("SPARSES/"+PREFIJO+"_MATRIZONA_RATINGS.pkl")

    print("Datos cargados...")

    #####################################################################

    if not loadData:
        print("Centroides generados...")

        modelUSR_RAT = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelUSR_RAT.fit(USR_RAT)

        model_USR_RAT_labels = modelUSR_RAT.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_RAT_labels", model_USR_RAT_labels)

    else:

        model_USR_RAT_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_RAT_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Usuarios RATING , Películas RAW")
    print("U_rat")
    IM1 = getIncertidumbre(Kusers, Kitems, VAL_TEST, model_USR_RAT_labels, model_DIFF_RAW_labels,
                           fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD_RATINGS_RAW_INC")
    print("-" * 65)


def methodVioNoVio(Kusers=100, Kitems=100, SEED=100, verbose=1, loadData=False, max_iter=100):
    # Cluster con los usuarios representados por la matrizona (vio o no vio)
    # En el artículo si se pone:
    # Crea clusters si no existen y calcula la incertidumbre del método

    USR_RAW, DIFF_RAW,  VAL_TRAIN, VAL_TEST = getMethodData()

    USR_VIO = getPickle("SPARSES/"+PREFIJO+"_MATRIZONA_VIO_NO_VIO.pkl")

    print("Datos cargados...")

    #####################################################################

    if not loadData:
        print("Centroides generados...")

        modelUSR_VIO = MiniBatchKMeans(Kusers, init='random', max_no_improvement=1000, batch_size=1024,
                                       verbose=verbose, random_state=SEED, max_iter=max_iter)

        modelUSR_VIO.fit(USR_VIO)

        model_USR_VIO_labels = modelUSR_VIO.labels_
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

        toPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_VIO_labels", model_USR_VIO_labels)

    else:

        model_USR_VIO_labels = getPickle("ClusteringResults/"+PREFIJO+"_model_" + str(Kusers) + "_USR_VIO_labels.pkl")
        model_DIFF_RAW_labels = getPickle("DIFFS/"+PREFIJO+"_BOOKS_RAW_LABELS_TEST.pkl")

    #####################################################################

    print("[" + str(Kusers) + " Clusters] Usuarios VIO NO VIO , Películas RAW")
    print("U_vio")
    IM1 = getIncertidumbre(Kusers, Kitems, VAL_TEST, model_USR_VIO_labels, model_DIFF_RAW_labels,
                           fileName="ClusteringResults/Inc/"+PREFIJO+"_METHOD_VIO_NO_VIO_RAW_INC")
    print("-" * 65)

########################################################################################################################
#  Llamadas
########################################################################################################################

#IM1 = getPickle("ClusteringResults/Inc/METHOD4.2_DAY1_VALPOND_INC")

#IM1 = getPickle("ClusteringResults/model_DAY1_100_VALPOND_RAW_labels")
#print collections.Counter(IM1).most_common(100)

# OJO no tengo los ficheros DIFF_RAW_LABELS_TEST originales del 1 al 6, así que no conozco los
# clusters de item/diferencias que se tenían cuando se obtuvieron los resultados del paper
# PUEDO generar estos archivos utilizando la función "getDIFFcentroids", pero ya no quedan los mismos clusters que
# se publicaron en el paper
# Los ficheros DIFF_* que hay ahora en la copia externa son regenerando los cluster

Kusers = 10
Kitems = 10  # tiene que ser el mismo valor que el que se utilizó para ML1M_MOVIES_RAW_{CENTROIDS,LABELS}_{TRAIN,TEST}
max_iter_kmeans = 1000
# getDIFFcentroids(Kitems, SEED=100, verbose=1, testData=True)  # para obtener ficheros DIFF_* con los clusters
# exit()
CARGA = True  # False => vuelve a hacer los k-means de usuarios
methodTwo(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo: Cl(Uraw)
methodThree(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo: Cl(Uraw + Uti)
methodFour(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo: Cl(U Uti)
methodFourDotTwo(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo: Cl(UW-Uti)
methodEMB(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo: Cl(Uemb)
methodRatings(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo:
methodVioNoVio(Kusers=Kusers, Kitems=Kitems, loadData=CARGA, max_iter=max_iter_kmeans)  # En el artículo:


# performAllTests(day=DAY)
