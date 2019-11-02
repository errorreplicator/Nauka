from keras.models import Sequential, Model
from keras.layers import Dense,Input,Flatten,concatenate,Embedding, Dropout


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

    model.add(Dense(128,input_shape=shape,activation='relu'))
    model.add(Dense(64,activation='relu'))
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
