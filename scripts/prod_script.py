import sys
import os
import click
import pandas as pd
from pathlib import Path
from fastai.tabular.all import *
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(package_dir))
from models.data_struct import *
from ML_Task.data_modeling import *
from ML_Task.data_exploratory import load_data


@click.command()
@click.option(
    "--model_path",
    type=str,
    default="../models/learner.pkt",
)
@click.option(
    "--class_path",
    type=str,
    default="../models/model.pt",
)
@click.option(
    "--train_data_path",
    type=str,
    default="../Data/facebook_train.csv",
)
@click.option(
    "--test_data_path",
    type=str,
    default="../Data/facebook_test_no_target.csv",
    required=True,
)
@click.option("--load_existant_model", default=True)
@click.option("--lr_algos", type=tuple, default=(minimum, steep, valley, slide))
@click.option("--layers", type=list, default=[200, 75, 5, 3])
@click.option("--metrics", type=list, default=[mse, rmse, mae])
@click.option("--n_outputs", type=int, default=3)
@click.option("--n_epochs", type=int, default=75)
@click.option("--valid_data", type=float, default=0.3)
@click.option("--export_learner_class_path", type=click.Tuple([str, str]), default=None)
@click.option("--export_preds_path", type=str, default="../outputs/")
# main function
def main(
    load_existant_model,
    train_data_path,
    test_data_path,
    model_path,
    class_path,
    lr_algos,
    layers,
    metrics,
    n_outputs,
    n_epochs,
    valid_data,
    export_preds_path,
    export_learner_class_path,
):
    """Wrap the main function."""
    main_(
        load_existant_model=load_existant_model,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        model_path=model_path,
        class_path=class_path,
        lr_algos=lr_algos,
        layers=layers,
        metrics=metrics,
        n_outputs=n_outputs,
        n_epochs=n_epochs,
        valid_data=valid_data,
        export_preds_path=export_preds_path,
        export_learner_class_path=export_learner_class_path,
    )


def main_(
    load_existant_model,
    train_data_path,
    test_data_path,
    model_path,
    class_path,
    lr_algos,
    layers,
    metrics,
    n_outputs,
    n_epochs,
    valid_data,
    export_preds_path,
    export_learner_class_path,
):
    """
    The primary function that enables users to train new models, load existing models, and forecast outcomes using new data.
    """
    # case loading existant model
    if load_existant_model:
        # load learner
        learner = load_learner(model_path)
        # load model class
        model_inst = torch.load(class_path)
    # case training a new model
    else:
        # load train data
        df_train = load_data(train_data_path)
        # check data sanity
        check_data_sanity(df_train, "TRAIN_DATA")
        # model instance
        model_inst = DataModeling()
        # fit the model
        learner = model_inst.train(
            df_train, layers, n_epochs, valid_data, lr_algos, metrics, n_outputs
        )

    # load new data
    df_test = load_data(test_data_path)
    # check data consistency
    check_data_sanity(df_test, "TEST_DATA")
    # predict
    preds = model_inst.test(df_test, learner)
    # export preds to csv
    if export_preds_path != None:
        Path(export_preds_path).mkdir(exist_ok=True)
        preds.to_csv(export_preds_path + "preds.csv", index=False)
    # export models
    if export_learner_class_path != None:
        learner_path, class_path = export_learner_class_path
        DataModeling.export(learner, model_inst, class_path, learner_path)
    # return preds
    return preds


def check_data_sanity(df, struct_section):
    """
    check data consistency
    """
    # get data structure
    data_struct = data_structure[struct_section]
    # case df is not empty
    if not df.empty:
        # case if number of columns are =
        if df.shape[1] == len(data_struct):
            # case if the same columns
            if pd.Series(data_struct).isin(df.columns).all():
                print("Data is checked ...")
                print("Data is Valid")
            else:
                raise ValueError("Data Structure is not valid")
        else:
            raise ValueError("Missing rows for the given data")
    else:
        raise ValueError("Empty File")
    return False


if __name__ == "__main__":
    main()