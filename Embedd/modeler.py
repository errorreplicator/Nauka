from keras.models import Sequential, Model
from keras.layers import Dense,Input,Flatten,concatenate,Embedding, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import keras
def get_model_Emb1():
    # Inputs
    Workclass = Input(shape=(1,), name='Workclass')
    Education = Input(shape=(1,), name='Education')
    MaritalStatus = Input(shape=(1,), name='MaritalStatus')
    Occupation = Input(shape=(1,), name='Occupation')
    Relationship = Input(shape=(1,), name='Relationship')
    Race = Input(shape=(1,), name='Race')
    Sex = Input(shape=(1,), name='Sex')
    Country = Input(shape=(1,), name='Country')
    Numerical = Input(shape=(5,), name='Numerical')

    # Embeddigs
    Workclass_emb = Embedding(input_dim=9, output_dim=5, name='Workclass_emb')(Workclass)
    Education_emb = Embedding(input_dim=26, output_dim=8, name='Education_emb')(Education)
    MaritalStatus_emb = Embedding(input_dim=7, output_dim=3, name='MaritalStatus_emb')(MaritalStatus)
    Occupation_emb = Embedding(input_dim=15, output_dim=7, name='Occupation_emb')(Occupation)
    Relationship_emb = Embedding(input_dim=6, output_dim=3, name='Relationship_emb')(Relationship)
    Race_emb = Embedding(input_dim=5, output_dim=2, name='Race_emb')(Race)
    Sex_emb = Embedding(input_dim=2, output_dim=2, name='Sex_emb')(Sex)
    Country_emb = Embedding(input_dim=42, output_dim=21, name='Country_emb')(Country)

    concat_emb = concatenate([
        Flatten()(Workclass_emb)
        , Flatten()(Education_emb)
        , Flatten()(MaritalStatus_emb)
        , Flatten()(Occupation_emb)
        , Flatten()(Relationship_emb)
        , Flatten()(Race_emb)
        , Flatten()(Sex_emb)
        , Flatten()(Country_emb)
    ])

    numerical = Dense(128, activation='relu')(Numerical)

    concat_all = concatenate([
        concat_emb
        , numerical
    ])

    main = Dense(128, activation='relu')(concat_all)
    main = Dense(64, activation='relu')(main)
    output = Dense(1, activation='sigmoid')(main)

    model = Model(
        inputs=[Workclass, Education, MaritalStatus, Occupation, Relationship, Race, Sex, Country, Numerical],
        outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_Emb1Dropout():
    # Inputs
    Workclass = Input(shape=(1,), name='Workclass')
    Education = Input(shape=(1,), name='Education')
    MaritalStatus = Input(shape=(1,), name='MaritalStatus')
    Occupation = Input(shape=(1,), name='Occupation')
    Relationship = Input(shape=(1,), name='Relationship')
    Race = Input(shape=(1,), name='Race')
    Sex = Input(shape=(1,), name='Sex')
    Country = Input(shape=(1,), name='Country')
    Numerical = Input(shape=(5,), name='Numerical')

    # Embeddigs
    Workclass_emb = Embedding(input_dim=9, output_dim=5, name='Workclass_emb')(Workclass)
    Education_emb = Embedding(input_dim=26, output_dim=8, name='Education_emb')(Education)
    MaritalStatus_emb = Embedding(input_dim=7, output_dim=3, name='MaritalStatus_emb')(MaritalStatus)
    Occupation_emb = Embedding(input_dim=15, output_dim=7, name='Occupation_emb')(Occupation)
    Relationship_emb = Embedding(input_dim=6, output_dim=3, name='Relationship_emb')(Relationship)
    Race_emb = Embedding(input_dim=5, output_dim=2, name='Race_emb')(Race)
    Sex_emb = Embedding(input_dim=2, output_dim=2, name='Sex_emb')(Sex)
    Country_emb = Embedding(input_dim=42, output_dim=21, name='Country_emb')(Country)

    concat_emb = concatenate([
        Flatten()(Workclass_emb)
        , Flatten()(Education_emb)
        , Flatten()(MaritalStatus_emb)
        , Flatten()(Occupation_emb)
        , Flatten()(Relationship_emb)
        , Flatten()(Race_emb)
        , Flatten()(Sex_emb)
        , Flatten()(Country_emb)
    ])

    numerical = Dense(128, activation='relu')(Numerical)

    concat_all = concatenate([
        concat_emb
        , numerical
    ])

    main = Dropout(0.2)(concat_all)
    main = Dense(128, activation='relu')(main)
    main = Dropout(0.1)(main)
    main = Dense(64, activation='relu')(main)
    main = Dropout(0.1)(main)
    output = Dense(1, activation='sigmoid')(main)

    model = Model(
        inputs=[Workclass, Education, MaritalStatus, Occupation, Relationship, Race, Sex, Country, Numerical],
        outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_Emb2Dropout(): #Sex not embedded
    # Inputs
    Workclass = Input(shape=(1,), name='Workclass')
    Education = Input(shape=(1,), name='Education')
    MaritalStatus = Input(shape=(1,), name='MaritalStatus')
    Occupation = Input(shape=(1,), name='Occupation')
    Relationship = Input(shape=(1,), name='Relationship')
    Race = Input(shape=(1,), name='Race')
    Country = Input(shape=(1,), name='Country')
    Numerical = Input(shape=(6,), name='Numerical')

    # Embeddigs
    Workclass_emb = Embedding(input_dim=9, output_dim=5, name='Workclass_emb')(Workclass)
    Education_emb = Embedding(input_dim=26, output_dim=8, name='Education_emb')(Education)
    MaritalStatus_emb = Embedding(input_dim=7, output_dim=3, name='MaritalStatus_emb')(MaritalStatus)
    Occupation_emb = Embedding(input_dim=15, output_dim=7, name='Occupation_emb')(Occupation)
    Relationship_emb = Embedding(input_dim=6, output_dim=3, name='Relationship_emb')(Relationship)
    Race_emb = Embedding(input_dim=5, output_dim=2, name='Race_emb')(Race)

    Country_emb = Embedding(input_dim=42, output_dim=21, name='Country_emb')(Country)

    concat_emb = concatenate([
        Flatten()(Workclass_emb)
        , Flatten()(Education_emb)
        , Flatten()(MaritalStatus_emb)
        , Flatten()(Occupation_emb)
        , Flatten()(Relationship_emb)
        , Flatten()(Race_emb)
        , Flatten()(Country_emb)
    ])

    numerical = Dense(128, activation='relu')(Numerical)

    concat_all = concatenate([
        concat_emb
        , numerical
    ])

    main = Dropout(0.2)(concat_all)
    main = Dense(128, activation='relu')(main)
    main = Dropout(0.1)(main)
    main = Dense(64, activation='relu')(main)
    main = Dropout(0.1)(main)
    output = Dense(1, activation='sigmoid')(main)

    model = Model(
        inputs=[Workclass, Education, MaritalStatus, Occupation, Relationship, Race, Country, Numerical],
        outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_Seq(shape):
    model = Sequential()

    model.add(Dense(1024,input_shape=shape,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

def get_model_SeqDrop(shape):
    model = Sequential()

    model.add(Dense(128,input_shape=shape,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

def get_model_Emb1DropoutBIG():
    # Inputs
    Workclass = Input(shape=(1,), name='Workclass')
    Education = Input(shape=(1,), name='Education')
    MaritalStatus = Input(shape=(1,), name='MaritalStatus')
    Occupation = Input(shape=(1,), name='Occupation')
    Relationship = Input(shape=(1,), name='Relationship')
    Race = Input(shape=(1,), name='Race')
    Sex = Input(shape=(1,), name='Sex')
    Country = Input(shape=(1,), name='Country')
    Numerical = Input(shape=(5,), name='Numerical')

    # Embeddigs
    Workclass_emb = Embedding(input_dim=9, output_dim=5, name='Workclass_emb')(Workclass)
    Education_emb = Embedding(input_dim=26, output_dim=8, name='Education_emb')(Education)
    MaritalStatus_emb = Embedding(input_dim=7, output_dim=3, name='MaritalStatus_emb')(MaritalStatus)
    Occupation_emb = Embedding(input_dim=15, output_dim=7, name='Occupation_emb')(Occupation)
    Relationship_emb = Embedding(input_dim=6, output_dim=3, name='Relationship_emb')(Relationship)
    Race_emb = Embedding(input_dim=5, output_dim=2, name='Race_emb')(Race)
    Sex_emb = Embedding(input_dim=2, output_dim=2, name='Sex_emb')(Sex)
    Country_emb = Embedding(input_dim=42, output_dim=21, name='Country_emb')(Country)

    concat_emb = concatenate([
        Flatten()(Workclass_emb)
        , Flatten()(Education_emb)
        , Flatten()(MaritalStatus_emb)
        , Flatten()(Occupation_emb)
        , Flatten()(Relationship_emb)
        , Flatten()(Race_emb)
        , Flatten()(Sex_emb)
        , Flatten()(Country_emb)
    ])

    numerical = Dense(128, activation='relu')(Numerical)

    concat_all = concatenate([
        concat_emb
        , numerical
    ])

    # main = Dropout(0.2)(concat_all)
    main = Dense(128, activation='relu')(concat_all)
    main = Dropout(0.1)(main)
    main = Dense(512, activation='relu')(main)
    main = Dropout(0.1)(main)
    main = Dense(128, activation='relu')(main)
    main = Dropout(0.1)(main)
    output = Dense(1, activation='sigmoid')(main)

    model = Model(
        inputs=[Workclass, Education, MaritalStatus, Occupation, Relationship, Race, Sex, Country, Numerical],
        outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def tester():
    Categorical = Input(shape=(51,), name='Categorical')
    Numerical = Input(shape=(5,), name='Numerical')

    Mid_Categorical = Dense(1024, activation='relu')(Categorical)

    concat = concatenate([
        Mid_Categorical
        ,Numerical
    ])

    main = Dense(1024, activation='relu')(concat)
    main = Dropout(0.1)(main)
    main = Dense(512, activation='relu')(main)
    main = Dropout(0.1)(main)
    main = Dense(128, activation='relu')(main)
    main = Dropout(0.1)(main)
    output = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=[Categorical,Numerical],outputs=output)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model


def evaluateSeqModel(X_test, y_test, model, name):
    val = model.evaluate(X_test, y_test)
    pred = model.predict(X_test)
    pred_class = model.predict_classes(X_test)
    proba = model.predict_proba(X_test)
    report = classification_report(y_test, pred_class)
    cm = confusion_matrix(y_true=y_test, y_pred=pred_class)
    df = pd.DataFrame({'y_test': y_test.flatten(), 'y_predi': pred_class.flatten(), 'y_proba': pred.flatten()})
    print(df.head())
    # df.sort_values('y_proba', axis=0, ascending=False, inplace=True)
    print('Validaiton:')
    print(val)
    print('Clasifcation report:')
    print(report)
    print('Confiuzino Matrix:')
    print(cm)

    filename = f'{name}.txt'
    # file = open(f'/home/piotr/data/test/{filename}','w')
    with open(f'/home/piotr/data/test/{filename}','w') as file:
        file.write('Validation:')
        file.write('\n')
        file.write(str(val) + '\n\n')
        file.write('Clasificaton report:')
        file.write('\n')
        file.write(str(report) + '\n\n')
        file.write('Confiuzion Matrix')
        file.write('\n')
        file.write(str(cm) + '\n\n')
        file.write('Prediction Classes')
        file.write('\n')
        file.write(df.to_string() + '\n\n')

    # return y_test,pred_class,pred


def evaluateFunModel(X_test, y_test, model, name):
    val = model.evaluate(X_test, y_test)
    pred = model.predict(X_test)
    pred_class = np.where(pred>=0.5,1,0)
    report = classification_report(y_test, pred_class)
    cm = confusion_matrix(y_true=y_test, y_pred=pred_class)
    df = pd.DataFrame({'y_test': y_test.flatten(), 'y_predi': np.array(pred_class).flatten(), 'y_proba': np.array(pred).flatten()})
    # df.sort_values('y_proba', axis=0, ascending=False, inplace=True)
    print('Validaiton:')
    print(val)
    print('Clasifcation report:')
    print(report)
    print('Confiuzino Matrix:')
    print(cm)

    filename = f'{name}.txt'
    # file = open(f'/home/piotr/data/test/{filename}','w')
    with open(f'/home/piotr/data/test/{filename}','w') as file:
        file.write('Validation:')
        file.write('\n')
        file.write(str(val) + '\n\n')
        file.write('Clasificaton report:')
        file.write('\n')
        file.write(str(report) + '\n\n')
        file.write('Confiuzion Matrix')
        file.write('\n')
        file.write(str(cm) + '\n\n')
        file.write('Prediction Classes')
        file.write('\n')
        file.write(df.to_string() + '\n\n')










