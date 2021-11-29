- 👋 Hi, I’m @swathi789200
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
swathi789200/swathi789200 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<html><h1><center style=\"color:blue\">Applying LSTM Models on Raw Data</center></h1></html>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Libraries\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Import Keras\n",
    "from keras import backend as K\n",
    "from keras.models import Sequential\n",
    "from keras.layers import LSTM\n",
    "from keras.layers.core import Dense, Dropout\n",
    "from keras.layers import BatchNormalization\n",
    "from keras.regularizers import L1L2\n"
   ]
  },
  {
  "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Activities are the class labels\n",
    "# It is a 6 class classification\n",
    "ACTIVITIES = {\n",
    "    0: 'WALKING',\n",
    "    1: 'WALKING_UPSTAIRS',\n",
    "    2: 'WALKING_DOWNSTAIRS',\n",
    "    3: 'SITTING',\n",
    "    4: 'STANDING',\n",
    "    5: 'LAYING',\n",
     "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# function to print the confusion matrix\n",
    "\n",
    "def confusion_matrix(Y_true, Y_pred):\n",
    "    \n",
    "    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])\n",
    "    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])\n",
    "\n",
     "    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])\n",
    "\n",
    "    \n",
    "   # result = confusion_matrix(Y_true, Y_pred)\n",
    "\n",
    "    #plt.figure(figsize=(10, 8))\n",
    "   # sns.heatmap(result, \n",
    "    #            xticklabels= list(ACTIVITIES.values()), \n",
    "     #           yticklabels=list(ACTIVITIES.values()), \n",
    "      #          annot=True, fmt=\"d\");\n",
    "   # plt.title(\"Confusion matrix\")\n",
    "   # plt.ylabel('True label')\n",
    "   # plt.xlabel('Predicted label')\n",
    "    plt.show()  "
   ]
    },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<html><h1><p style=\"color:red\">Loading Data / Making Data </p></h1></html>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
    "metadata": {},
   "outputs": [],
   "source": [
    "# Data directory\n",
    "DATADIR = 'UCI_HAR_Dataset'\n",
    "\n",
    "# Raw data signals\n",
    "# Signals are from Accelerometer and Gyroscope\n",
    "# The signals are in x,y,z directions\n",
    "# Sensor signals are filtered to have only body acceleration\n",
    "# excluding the acceleration due to gravity\n",
    "# Triaxial acceleration from the accelerometer is total acceleration\n",
    "SIGNALS = [\n",
    "    \"body_acc_x\",\n",
    "    \"body_acc_y\",\n",
    "    \"body_acc_z\",\n",
    "    \"body_gyro_x\",\n",
    "    \"body_gyro_y\",\n",
    "    \"body_gyro_z\",\n",
    "    \"total_acc_x\",\n",
    "    \"total_acc_y\",\n",
    "    \"total_acc_z\"\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# function to read the data from csv file\n",
    "def _read_csv(filename):\n",
    "    return pd.read_csv(filename, delim_whitespace=True, header=None)\n",
    "\n",
    "# function to load the load\n",
    "def load_signals(subset):\n",
    "    signals_data = []\n",
    "\n",
    "    for signal in SIGNALS:\n",
    "        filename = f'UCI_HAR_Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'\n",
    "        signals_data.append(\n",
    "            _read_csv(filename).to_numpy()\n",
    "        ) \n",
    "\n",
    "    # Transpose is used to change the dimensionality of the output,\n",
    "    # aggregating the signals by combination of sample/timestep.\n",
    "    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)\n",
    "    return np.transpose(signals_data, (1, 2, 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_y(subset):\n",
    "    \"\"\"\n",
    "    The objective that we are trying to predict is a integer, from 1 to 6,\n",
    "    that represents a human activity. We return a binary representation of \n",
    "    every sample objective as a 6 bits vector using One Hot Encoding\n",
    "    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)\n",
    "    \"\"\"\n",
    "    filename = f'UCI_HAR_Dataset/{subset}/y_{subset}.txt'\n",
    "    y = _read_csv(filename)[0]\n",
    "\n",
    "    return pd.get_dummies(y).to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_data():\n",
    "    \"\"\"\n",
    "    Obtain the dataset from multiple files.\n",
    "    Returns: X_train, X_test, y_train, y_test\n",
    "    \"\"\"\n",
    "    X_train, X_test = load_signals('train'), load_signals('test')\n",
    "    y_train, y_test = load_y('train'), load_y('test')\n",
    "\n",
    "    return X_train, X_test, y_train, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing tensorflow\n",
    "np.random.seed(42)\n",
    "import tensorflow as tf\n",
    "tf.random.set_seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initializing parameters\n",
    "epochs = 30\n",
    "batch_size = 16\n",
    "n_hidden = 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "#function to count the number of classes\n",
    "def _count_classes(y):\n",
    "    return len(set([tuple(category) for category in y]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the train and test data\n",
    "X_train, X_test, Y_train, Y_test = load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "128\n",
      "9\n",
      "7352\n"
     ]
    }
   ],
   "source": [
    "timesteps = len(X_train[0])\n",
    "input_dim = len(X_train[0][0])\n",
    "n_classes = _count_classes(Y_train)\n",
    "\n",
    "print(timesteps)\n",
    "print(input_dim)\n",
    "print(len(X_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<html><h1><p style=\"color:red\">1. Defining the Architecture of 1-Layer of LSTM </p></h1></html>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "lstm_1 (LSTM)                (None, 32)                5376      \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 32)                0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 6)                 198       \n",
      "=================================================================\n",
      "Total params: 5,574\n",
      "Trainable params: 5,574\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Initiliazing the sequential model\n",
    "model = Sequential()\n",
    "# Configuring the parameters\n",
    "model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))\n",
    "# Adding a dropout layer\n",
    "model.add(Dropout(0.5))\n",
    "# Adding a dense output layer with sigmoid activation\n",
    "model.add(Dense(n_classes, activation='sigmoid'))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compiling the model\n",
    "model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
   "scrolled": false
   },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x131cf002fd0>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Training the model\n",
    "model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test),epochs=epochs)"
   ]
  },
  {
  "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Pred</th>\n",
       "      <th>LAYING</th>\n",
       "      <th>SITTING</th>\n",
       "      <th>STANDING</th>\n",
       "      <th>WALKING</th>\n",
       "      <th>WALKING_DOWNSTAIRS</th>\n",
       "      <th>WALKING_UPSTAIRS</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>True</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LAYING</th>\n",
       "      <td>513</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SITTING</th>\n",
       "      <td>0</td>\n",
       "      <td>418</td>\n",
       "      <td>70</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>STANDING</th>\n",
       "      <td>0</td>\n",
       "      <td>109</td>\n",
       "      <td>423</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>WALKING</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>461</td>\n",
       "      <td>14</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>WALKING_DOWNSTAIRS</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>377</td>\n",
       "      <td>42</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>WALKING_UPSTAIRS</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>462</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Pred                LAYING  SITTING  STANDING  WALKING  WALKING_DOWNSTAIRS  \\\n",
       "True                                                                         \n",
       "LAYING                 513        0        19        4                   0   \n",
       "SITTING                  0      418        70        0                   0   \n",
       "STANDING                 0      109       423        0                   0   \n",
       "WALKING                  0        1         1      461                  14   \n",
       "WALKING_DOWNSTAIRS       0        0         0        1                 377   \n",
       "WALKING_UPSTAIRS         0        0         0        7                   2   \n",
       "\n",
       "Pred                WALKING_UPSTAIRS  \n",
       "True                                  \n",
       "LAYING                             1  \n",
       "SITTING                            3  \n",
       "STANDING                           0  \n",
       "WALKING                           19  \n",
       "WALKING_DOWNSTAIRS                42  \n",
       "WALKING_UPSTAIRS                 462  "
      ]
     },
      "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Confusion Matrix\n",
    "new_confusion_matrix(Y_test, model.predict(X_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
    "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2947/2947 [==============================] - 2s 568us/step\n",
      "\n",
      "   cat_crossentropy  ||   accuracy \n",
      "  ____________________________________\n",
      "[0.47147004155789973, 0.9005768299102783]\n"
     ]
    }
   ],
   "source": [
    "score = model.evaluate(X_test, Y_test)\n",
    "\n",
    "print(\"\\n   cat_crossentropy  ||   accuracy \")\n",
    "print(\"  ____________________________________\")\n",
    "print(score)"
   ]
  },
