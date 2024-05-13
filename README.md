# Spark ML Homework

## Introduction

This project is a homework at the EPAM Data Engineering Mentor program. The main idea behind this task is to provide with a basic understanding of data analysis, as well as practical skills in using the framework to visualize data and uncover valuable insights.

This task does not not require own programming work, it is mainly about running the cells in the pre-written notebook to create a Spark ML job using Azure Databricks, and observing the results. Infrastructure should be set up using Terraform. The original copyright belongs to [EPAM](https://www.epam.com/). 

#### Some original instructions about the task:

* Deploy infrastructure with terraform
* Copy notebook and data into Databricks cluster
* Execute all the steps from "ML End-to-End Example" notebook
* Import data from your local machine into the Databricks File System (DBFS)
* Visualize the data using Seaborn and matplotlib
* Run a parallel hyperparameter sweep to train machine learning models on the dataset
* Explore the results of the hyperparameter sweep with MLflow
* Register the best performing model in MLflow
* Apply the registered model to another dataset using a Spark UDF
* Set up model serving for low-latency requests
* Expected results
  - Repository with notebook (with output results), configuration scripts, application sources, analysis etc.
  - Upload in task Readme MD file with link on repo, fully documented homework with screenshots and comments.


## About the repo

This repo is hosted [here](https://github.com/devtpc/Spark03-ML)

> [!NOTE]
> Some sensitive files, like configuration files, API keys, tfvars are not included in this repo.



## Problematic issues

### Data files

The original link to the data files is not working.

![Data file missing](/screenshots/img_dataset_not_found.png)

Solution:
The dataset, or at least a very similar-looking dataset is available from an alternative link

![Data file missing 2](/screenshots/img_dataset_alternative_link.png)

The dataset is also available in Databricks samples:

![Data file missing 3](/screenshots/img_dataset_databricks_sample.png)


### Provisioning the cluster with Terraform

The infrastructure can be setup with Terraform, however creating the cluster is problematic using Azure free tier, because of its limitations. A more detailed explanation is [here](https://github.com/devtpc/Spark02-SQL/tree/0c372d666202436aeea28eac6017de62eb62b8f9/texts/limitations)

Solution:
Create the cluster manually


### Selecting the right Databricks Runtime Version

When selecting the Databricks Runtime, there're two options: Standard and ML. An ML version should be selected, because only these contain the necessary libraries.

![Databricks Runtime 1](/screenshots/img_databrics_runtime_1.png)

While running the XGBoost model in newer runtime versions, an AttributeError arises

![Attribute Error](/screenshots/img_randomstate_error.png)

This problem arises, because the `RandomState` object is deprecated in the new versions of Python/NumPy. The easiest way to overcome this issue in a **test** environment is to change the runtime version back to an older one, like 9.1 LTS ML. In a **production** environment the function itself should be changed to using `numpy.random.Generator` instead of `RandomState`

### Set up model serving for low-latency requests

The last task is to set up model serving for low-latency requests. However, this feature is available only in the Premium and Enterprise workspaces, and thus - as agreed during the consultation - it is not in the scope of this task.

![Serving endpoint error](/screenshots/img_serving_endpoint_error.png)


## Creating the infrastructure

If you didn't destroy your Databricks infrastructure since the previous [Spark SQL Task](https://github.com/devtpc/Spark02-SQL/), the same infrastructure can be used. Note, that in that case you might need to change the [Databricks Runtime Environment](#selecting-the-right-databricks-runtime-version)

### Setup the configuration

Go to the [configcripts folder](/configscripts/) and copy/rename the `config.conf.template` file to `config.conf`. Change the AZURE_BASE, AZURE_LOCATION, and other values as instructed within the file.

In the [configcripts folder](/configscripts/) copy/rename the `terraform_backend.conf.template` file to `terraform_backend.conf`. Fill the parameters with the terraform data.

Propagate your config data to other folders with the [refreshconfs.sh](/configscripts/refresh_confs.sh) script, or with `make refresh-confs` from the main folder

The details are in comments in the config files.


### Creating the Databricks environment

Before starting, make sure, that config files were correctly set up and broadcasted in the 'Setup your configuration'. There should be a terraform.auto.tvars file with your azure config settings, and a backend.conf file with the backend settings in the terraform folder.

Log in to Azure CLI with `az login`


#### Setup the Azure base infrastructure with terraform

Use `make createinfra` command to create your azure infrastructure. Alternatively, to do it step-by-step:

In your terraform folder:
```
#initialize your terraform
terraform init --backend-config=backend.conf

#plan the deployment
terraform plan -out terraform.plan

#confirm and apply the deployment. If asked, answer yes.
terraform apply terraform.plan
```


#### Create a Databricks cluster manually

To create the databricks cluster, manually create the cluster at the 'Compute' screen. Take care to select the right [Databricks Runtime Environment](#selecting-the-right-databricks-runtime-version) as specified earlier.

![Create Cluster 1](/screenshots/img_cluster_create.png)


#### Create a Databricks cluster with Terraform


> [!NOTE]
> The cluster can be set up with Terraform as well, however the Azure free tier's limitations did not make it possibe for me to set it up this way. A more detailed explanation is [here](https://github.com/devtpc/Spark02-SQL/tree/0c372d666202436aeea28eac6017de62eb62b8f9/texts/limitations)



## Running the app

In the [Notebooks folder](/notebooks) there are 3 files:
* The [original notebook](/notebooks/ML_End_to_End_Example.dbc), which can be uploaded as a source to the Azure Databricks portal.
* The [solved notebook](/notebooks/ML_End_to_End_Example_solved.dbc), which contains the result of the runs and the visualizations. This can be viewed after uploading to the Azure Databricks portal.
* The [solved notebook as HTML version](https://devtpc.github.io/Spark03-ML/notebooks/ml_end_to_end_examlpe_solved.html), which can be viewed in the browser directly, without using Databricks.

To run the app, upload one of the first two notebooks to the Databricks account, and run it cell-by-cell.

### The code

As the code cells are well documented, this readme focuses mainly on the results, without the code.

### Loading the data

As the [source data location was problematic](#data-files), the option to load the data from the databricks sample datasets was choosen.

![Data selection](/screenshots/img_dataset_databricks_sample.png)

### Data exploration

After loading and merging the dataset, we can view the dataset structure by watching the first elements using the `head()` function.

![Data explore head](/screenshots/img_explore_head.png)

We visualize the histogram of the quality data using the seaborn library. The data seems to be normally distributed between 3 and 9.

![Data explore hist](/screenshots/img_explore_hist.png)

We define a wine as high quality if it has quality >= 7. We create boxplots to visualize the correlation between the quality and the other factors in the database.

![Data explore box](/screenshots/img_explore_box.png)

It seems obvious, that alcohol is highly correlated with quality, as the median for quality wines are at much higher level, than the 3rd quantile for the bad-quality wines. On the other hand, density seems to be an important factor, too, negatively correlating with quality.

We can check, that there are no null-values in our data.

![Data explore box](/screenshots/img_preprocessing_1.png)

### Baseline model


As the first step, we are creating a simple Random Forest Classifier without tuning the hyperparameters. It can be checked, that according to the model, which parameters are the most important in determining the quality:

![Simple model features](/screenshots/img_simple_model_features.png)

The AUC Value (Area under the ROC curve) can be seen on the UI, selecting the Experiments tab. The AUC value is 0.89, which is pretty decent.

![Simple model AUC](/screenshots/img_simple_auc.png)

### Using the model Registry

We register the model to Model Registry. The registration is both written in the console/cell window of the notebook, and on the UI.

![Simple model registered](/screenshots/img_simple_model_registered.png)

![Simple model registered UI](/screenshots/img_simple_model_registered_UI.png)

![Simple model registered UI2](/screenshots/img_simple_model_registered_UI2.png)


As we transition the model to "Production" stage, we can use the model using its path. It can be simply checked by writing out the AUC.

![Simple model transit to production](/screenshots/img_transit_production_1.png)

![Simple model transit to production 2](/screenshots/img_transit_production_2.png)


### Experimenting with a new model

The notebook contains a code which uses the xgboost library to train a more accurate model. It runs a parallel hyperparameter sweep to train multiple models in parallel, using Hyperopt and SparkTrials. As before, the code tracks the performance of each parameter configuration with MLflow.

This code runs pretty slow, as it sweeps through 96 different evaluations. However, at the end, the winner model's AUC is better, than the previous, simple one. The different model can be seen on the right side of the UI

![Hyperparameter tuning](/screenshots/img_bigrun.png)

> [!NOTE]
> The result best AUC is 0.923, which is better than the baseline model. According to the original task, the AUC should be 0.91 (or as per the charts, 0.9147), meaning better than the baseline model, however a bit worse, than our model calculated. This diversion from the original example is a bit unexpected, especially, that although the modelling uses randomization, we are using a seed.
> However, it is a very possible explanation, that something was changed in the xgboost model's implementation, or in those default values, which our model does not exactly set, and in which it relies on the default values. Accoording to the introductury part of the notebook, it was designed when 7.3 LTS ML runtime was widespread. Nowadays 9.1 LTS ML version is the oldest, which can be choosen in a Databricks cluster. As our result is better than the original one, it should not bother us.

From the experiments page we can compare the best resulting models and their hyperparameters:


![Hyperparameter tuning 2](/screenshots/img_bigrun_charts.png)

![Hyperparameter tuning 3](/screenshots/img_bigrun_charts_2.png)


### Save the best model

Using the code in the workbook, we save the best model in the MLFlow registry, and set it as the new production model. We can check on the UI, that now our new model is the production model:

![Set new model as production](/screenshots/img_bigrun_archieved.png)

It can be checked, that the production model really is our new best model, as it has the same AUC as we saw before:

![Set new model as production](/screenshots/img_bestproduction_auc.png)

### Use the model for prediction

The example notebook uses the train data to demonstrate the usage of the model, but it can be used on any new data, like data coming from a delta table. The coding idea behind the application is like this:
```
# wrap the model in a udf:
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# set the input columns structure:
udf_inputs = struct(*(X_train.columns.tolist()))

# load the new data:
new_data = spark.read.format("delta").load(table_path)


# apply the model to the new data:
new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

```
As it can be seen on the image, the model created prediction values between 0 and 1. Generally 50% can be used as a divider between the "final" 0 or 1 decision, however, when the cost of false negative and false positive largely differs, the threshold can be set differently.

![Prediction](/screenshots/img_prediction.pngg)

### Model serving for low-latency requests

The last task is to set up model serving for low-latency requests. However, this feature is available only in the Premium and Enterprise workspaces, and thus - as agreed during the consultation - it is not in the scope of this task.

![Serving endpoint error](/screenshots/img_serving_endpoint_error.png)

## Destroy the workspace using terraform

Run `make destroy-databricks-ws` to destroy the Databricks workspace.


