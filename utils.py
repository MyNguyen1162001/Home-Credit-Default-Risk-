import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


def visualize_category_features(
    dataset: pd.DataFrame,
    cat_columns: List[str],
    target = str
) -> None:
    # Calculate number of rows needed (2 plots per row)
    n_rows = (len(cat_columns) + 1) // 2  # Using ceiling division

    # Create subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    # Loop through each categorical column
    for idx, col in enumerate(cat_columns):
        sns.countplot(data=dataset, x=col, hue=target, ax=axes[idx])
        axes[idx].set_title(f'{col} by {target}')
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove empty subplots if any
    for idx in range(len(cat_columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    

def visualize_numerical_features(
    dataset: pd.DataFrame,
    num_columns: List[str],
    target = str
)-> None:
# Create subplots for each numerical column
    fig, axes = plt.subplots(
        len(num_columns), 
        1, 
        figsize=(12, 4*len(num_columns))
    )

    # Loop through each numerical column
    for idx, col in enumerate(num_columns):
        # Create distribution plot using seaborn
        sns.histplot(data=dataset, x=col, hue=target, multiple="dodge", ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col} by {target}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Alternative: Using KDE (Kernel Density Estimation) plots
    plt.figure(figsize=(12, 4*len(num_columns)))

    for idx, col in enumerate(num_columns):
        plt.subplot(len(num_columns), 1, idx+1)
        sns.kdeplot(data=dataset, x=col, hue=target,bw_adjust=0.5)
        plt.title(f'Density Distribution of {col} by {target}')
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Boxplots for another perspective
    plt.figure(figsize=(12, 4*len(num_columns)))

    for idx, col in enumerate(num_columns):
        plt.subplot(len(num_columns), 1, idx+1)
        sns.boxplot(data=dataset, x=target, y=col)
        plt.title(f'Boxplot of {col} by {target}')

    plt.tight_layout()
    plt.show()


def plot_categorical_violin(
        dataset : pd.DataFrame, 
        cat_column : List[str], 
        num_column : str,
        target_column: str, 
        title = 'Distribution by Credit Risk',
        # colors : List[str] = ['#91c08b', '#8080ff']
) -> None:
    for category in cat_column:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=dataset, 
                    x=category,
                    y= num_column,
                    hue=target_column,
                    split=True,
                    inner='box',
                    linewidth=1.5,       
                    saturation=0.75)

        plt.title(f'{num_column} Distribution by {category.title()} and {target_column}')
        plt.xlabel(category.title())
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  


def plot_categorical_violin_not_default(
        dataset : pd.DataFrame, 
        cat_column : List[str], 
        num_column : str,
        target_column: str, 
        title = 'Distribution by Credit Risk',
        colors : List[str] = ['#91c08b', '#8080ff']
) -> None:
    for category in cat_column:
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(data=dataset, 
                    x=category,
                    y= num_column,
                    hue=target_column,
                    split=True,
                    inner='box',
                    palette=colors,
                    linewidth=1.5,       
                    saturation=1)
        
        for violin in ax.collections:
            if isinstance(violin, matplotlib.collections.PolyCollection):
                violin.set_edgecolor(violin.get_facecolor())

        plt.title(f'{num_column} Distribution by {category.title()} and {target_column}')
        plt.xlabel(category.title())
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  


def process_categorical_features (
        dataset = pd.DataFrame,
        categories_columns = List[str],
        fill_value  = str ,
        encoding_type = str == 'onehot'
) -> pd.DataFrame: 
    for col in categories_columns: 
        dataset[col] = dataset[col].replace(
                    ['nan', 'None', 'NaN', 'null', '', ' '], 
                    fill_value
                )
        dataset[col] = dataset[col].fillna('missing')

        if encoding_type == 'label':
            encoders = {}
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
        elif encoding_type == 'onehot':
        # Create dummy variables with drop_first=True to avoid multicollinearity
            df_encoded = pd.get_dummies(
                dataset[col], 
                columns=col,
                drop_first=True,
                prefix=col
            )
            dataset = pd.concat([
                dataset.drop(columns=col), 
                df_encoded
            ], axis=1)
    return dataset


def processing_numerical_columns(
        dataset: pd.DataFrame,
        numerical_columns: List[str],
        fill_strategy: str = 'zero',
        handle_outliers: bool = True,
        scaling : bool = True,
        scaling_type: str = 'standard'
) -> pd.DataFrame: 
    for col in numerical_columns: 
        if fill_strategy == 'zero':
            dataset[col] = dataset[col].fillna(0)
        if fill_strategy == 'mean':
            fill_value = dataset[col].mean()
            dataset[col] = dataset[col].fillna(fill_value)
        if fill_strategy == 'median':
            fill_value = dataset[col].median()
            dataset[col] = dataset[col].fillna(fill_value)
    if handle_outliers:
        for col in numerical_columns:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap outliers
            dataset[col] = dataset[col].clip(lower_bound, upper_bound)
    if scaling :
        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_type == 'robust':
            scaler = RobustScaler()
    return dataset


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sklearn.base import BaseEstimator 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def plot_threshold_analysis(
        model: BaseEstimator, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.DataFrame, np.ndarray], 
        target_label: int =1
) -> None :
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0, 1, 0.01)
    aucs = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, aucs, 'b-')
    plt.xlabel('Threshold')
    plt.ylabel('AUC')
    plt.title('AUC vs Classification Threshold')
    plt.grid(True)
    plt.legend()
    plt.show()


def find_optimal_threshold(
    model: BaseEstimator,          # scikit-learn model or pipeline
    X_test: Union[pd.DataFrame, np.ndarray],  # Features can be DataFrame or numpy array
    y_test: Union[pd.Series, np.ndarray],     # Target can be Series or numpy array
    current_accuracy: float = 0.75  # Float between 0 and 1
) -> Tuple[float, float]: 
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Create range of thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    
    # Store metrics for each threshold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype('int')
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axhline(y=current_accuracy, color='r', linestyle='--', label='Current Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find threshold that maximizes accuracy
    best_accuracy_idx = np.argmax(accuracies)
    best_threshold_accuracy = thresholds[best_accuracy_idx]
    
    # Find threshold that maximizes F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_f1_idx]
    
    print(f"Best threshold for accuracy: {best_threshold_accuracy:.3f}")
    print(f"Best accuracy score: {accuracies[best_accuracy_idx]:.3f}")
    print(f"\nBest threshold for F1: {best_threshold_f1:.3f}")
    print(f"Best F1 score: {f1_scores[best_f1_idx]:.3f}")
    
    return best_threshold_accuracy, best_threshold_f1


# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from typing import Dict, Union, Tuple, TypeVar
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_model_pipeline(
    model_name: str ='lr'
) -> Pipeline:
   models = {
       'lr': LogisticRegression(random_state=42),
       'rf': RandomForestClassifier(random_state=42),
        'xgb': XGBClassifier(
        use_label_encoder=False,  # Add this
        eval_metric='logloss',    # Add this
        random_state=42,
        enable_categorical=True    # Add this if you have categorical features
        ),
       'lgb': LGBMClassifier(random_state=42)
   }
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', models[model_name])
   ])
   
   return pipeline


def train_evaluate(
        X_train: Union[pd.DataFrame, np.ndarray], 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.DataFrame, np.ndarray], 
        model: BaseEstimator,
        threshold: float 
    ) -> Tuple[Dict[str, float], BaseEstimator]:
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype('int')
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, model


def cross_validate_models(
      X:Union[pd.DataFrame, np.ndarray], 
      y: Union[pd.DataFrame, np.ndarray],
      models: List[str] =['lr', 'rf', 'lgb']
) -> Dict[str, Dict[str, float]]:
   results: Dict[str, Dict[str, float]] = {}
   for model_name in models:
       pipeline = create_model_pipeline(model_name)
       scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
       results[model_name] = {
           'mean_auc': scores.mean(),
           'std_auc': scores.std()
       }
   return results


from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def split_train_test(
    dataset: pd.DataFrame,
    target_variable: str,
    test_size: float = 0.2,
    is_rebalance: bool = True,
    positive_num: int = None,
    negative_num: int = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # Split features and target
    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # print(y_train.value_counts())

    if is_rebalance:
        smote = SMOTE(
            sampling_strategy={1: positive_num, 0: negative_num},
            random_state=42
        )
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(y_train.value_counts())
        # Convert y_train to numpy array for consistency
        # y_train = np.array(y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test


def plot_credit_boxplot(
    dataset: pd.DataFrame,
    cat_column: List[str],
    num_column: str,
    target_column: str,
    colors: List[str] = ['#91c08b', '#ff9999'],  # Light green for good, light red for bad
    figsize_length : int = 12,
    figsize_width: int = 6
) -> None:
    plt.figure(figsize=(figsize_length, figsize_width))
    
    # Create boxplot with customization
    for category in cat_column:
        sns.boxplot(data=dataset,
                    x=category,
                    y=num_column,
                    hue=target_column,
                    palette=colors,
                    flierprops={'marker': 'o', 'markerfacecolor': None, 'markersize': 4},
                    boxprops={'alpha': 0.5},
                    whiskerprops={'linestyle': '-'},
                    medianprops={'color': 'black'},
                    showfliers=True)  # Show outlier points
        
        # Customize the plot
        plt.title(f'{num_column} Distribution by {category} and Credit Risk')
        plt.xlabel(f'{category}')
        plt.ylabel('Credit Amount (US Dollar)')
        
        # Customize legend
        plt.legend(title='', labels=['good', 'bad'], 
                bbox_to_anchor=(1, 1), loc='upper right')
        
        # Add gridlines
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def ScoreScalingParameters (
        points_to_double_odds=20, 
        ref_score=600, 
        ref_odds=50
):
    factor = points_to_double_odds / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)
    return factor, offset


def calculate_woe_iv(X, y, feature, bins=10):
   bins_dict = {}
   woe_dict = {}
   iv_dict = {}

   # Create bins based on data type
   if X[feature].dtype in ['object', 'category']:
       groups = pd.qcut(X[feature].astype('category').cat.codes, q=bins, duplicates='drop')
   else:
       groups = pd.qcut(X[feature], q=bins, duplicates='drop')
   
   bins_dict[feature] = groups.unique()
   
   # Calculate WOE and IV for each group
   grouped = pd.DataFrame({'group': groups, 'target': y}).groupby('group')
   iv = 0
   group_woes = {}
   
   for group in grouped.groups.keys():
       group_stats = grouped.get_group(group)
       good = sum(group_stats['target'] == 0) + 0.5  # Add smoothing
       bad = sum(group_stats['target'] == 1) + 0.5
       
       n_bad = sum(y == 0)
       n_good = sum(y == 1)
       
       eps = 1e-10
       good_rate = good / n_bad if n_bad != 0 else eps
       bad_rate = bad / n_good if n_good != 0 else eps
       
       woe = np.log(good_rate / bad_rate)
       iv += (good_rate - bad_rate) * woe
       group_woes[group] = woe
   
   woe_dict[feature] = group_woes
   iv_dict[feature] = iv
   
   return woe_dict, iv_dict, bins_dict


def transform_woe(
        woe_dict: pd.DataFrame,
        bins_dict: pd.DataFrame,
        X: pd.DataFrame,
        feature: str
):
    if X[feature].dtype in ['object', 'category']:
        groups = pd.qcut(X[feature].astype('category').cat.codes, 
                        q=len(bins_dict[feature]), 
                        duplicates='drop')
    else:
        groups = pd.qcut(X[feature], 
                        q=len(bins_dict[feature]), 
                        duplicates='drop')
        
    return groups.map(woe_dict[feature])


def fit(X, y):
    """Fit the scorecard model"""
    X_woe = pd.DataFrame()
    woe_dicts = {}
    
    # Calculate WOE and IV for each feature
    for feature in X.columns:
        woe_dict, iv, bins_dict = calculate_woe_iv(X, y, feature)
        woe_dicts[feature] = woe_dict
        X_woe[feature] = transform_woe(woe_dict=woe_dict, 
                                     bins_dict=bins_dict,
                                     X=X,
                                     feature=feature)
    
    # Convert categorical columns to string type first
    cat_cols = ['Age', 'Sex', 'Job', 'Housing', 'Purpose', 'Saving accounts', 'Checking account']
    for col in cat_cols:
        if col in X_woe.columns:
            X_woe[col] = X_woe[col].astype(str)
    
    # Process features
    X_woe = process_categorical_features(
        dataset=X_woe,
        categories_columns=cat_cols,
        fill_value='missing',
        encoding_type='label'
    )
    X_woe = processing_numerical_columns(
        dataset=X_woe,
        numerical_columns=['Credit amount', 'Duration', 'Age'],
        fill_strategy='mean',
        handle_outliers=False,
        scaling_type='robust'
    )
    
    # Fit model
    model = LogisticRegression(random_state=42)
    model.fit(X_woe, y)
    
    return model, woe_dicts, X_woe


def transform_to_score(model, X, X_woe, offset, factor):
    """Transform features to credit score"""
    
    # Calculate log odds
    log_odds = model.predict_proba(X_woe)[:, 1]
    log_odds = np.log(log_odds / (1 - log_odds))
    
    # Transform to score
    scores = offset + factor * log_odds
    return scores


def convert_days_to_years (
        dataset: pd.DataFrame,
        cols: List[str]
):
    for col in cols:
        dataset[f'{col}_converted'] = dataset[col].abs()/365
    return dataset 


def plot_income_distribution_combined(
        df: pd.DataFrame, 
        col: str, 
        target_col: str
):    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    print (Q1, Q3, IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_no_outliers = df[df[col].between(lower_bound, upper_bound)]
    df_millions = df.copy()
    df_millions[col] = df_millions[col].clip(lower_bound, upper_bound) / 1_000_000
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    sns.histplot(data=df, 
                x=np.log10(df[col]), 
                hue=target_col, 
                multiple="dodge",
                ax=axes[0,0])
    axes[0,0].set_title(f'Histogram of log10({col})')
    axes[0,0].set_xlabel(f'Log10({col})')
    
    sns.kdeplot(data=df, 
                x=np.log10(df[col]), 
                hue=target_col,
                fill=True,
                alpha=0.5,
                ax=axes[0,1])
    axes[0,1].set_title(f'KDE of log10({col})')
    axes[0,1].set_xlabel(f'Log10({col})')
    
    sns.histplot(data=df_no_outliers, 
                x=col, 
                hue=target_col, 
                multiple="dodge",
                ax=axes[1,0])
    axes[1,0].set_title(f'Histogram of {col} (Without Outliers)')
    axes[1,0].set_xlabel(f'{col}')
    
    sns.kdeplot(data=df_no_outliers, 
                x=col, 
                hue=target_col,
                fill=True,
                alpha=0.5,
                ax=axes[1,1])
    axes[1,1].set_title('KDE of Income (Without Outliers)')
    axes[1,1].set_xlabel(f'{col}')
    
    sns.histplot(data=df_millions, 
                x=col, 
                hue=target_col, 
                multiple="dodge",
                ax=axes[2,0])
    axes[2,0].set_title(f'Histogram of {col} (In Millions, Outliers Capped)')
    axes[2,0].set_xlabel(f'{col} (Millions)')
    
    sns.kdeplot(data=df_millions, 
                x=col, 
                hue=target_col,
                fill=True,
                alpha=0.5,
                ax=axes[2,1])
    axes[2,1].set_title(f'KDE of {col} (In Millions, Outliers Capped)')
    axes[2,1].set_xlabel(f'{col} (Millions)')
    
    plt.tight_layout()
    plt.show()
    
    print("\nSummary Statistics:")
    print(f"Original {col} Range: {df[col].min():,.0f} to {df[col].max():,.0f}")
    print(f"{col} Range (Without Outliers): {lower_bound:,.0f} to {upper_bound:,.0f}")


def create_multiple_ratios(
    dataset: pd.DataFrame,
    column_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    
    df = dataset.copy()
    
    for numerator, denominator in column_pairs:
        # Create new column name
        new_column = f'{numerator}_per_{denominator}'
        
        # Calculate ratio, handling potential division by zero or null values
        df[new_column] = df[numerator].div(df[denominator]).fillna(0)
        
    return df


def analyze_categorical_features(dataset: pd.DataFrame, 
                               category_column: str,
                               target_column: str = 'TARGET',
                               plot_type: str = 'both',
                               figsize: tuple = (12, 6)) -> None:
    
    total_counts = dataset[category_column].value_counts()
    bad_counts = dataset[dataset[target_column] == 1][category_column].value_counts()
    proportions = (bad_counts / total_counts * 100).sort_values(ascending=False)
    
    summary_df = pd.DataFrame({
        'Total_Count': total_counts,
        'Bad_Count': bad_counts,
        'Bad_Proportion': proportions
    }).round(2)
    
    summary_df = summary_df.sort_values('Bad_Proportion', ascending=False)
    
    if plot_type in ['both', 'proportion']:
        plt.figure(figsize=figsize)
        ax1 = plt.gca()
        
        summary_df['Bad_Proportion'].plot(kind='bar', color='darkred', ax=ax1)
        ax1.set_title(f'Bad Customer Proportion by {category_column}')
        ax1.set_ylabel('Bad Customer Proportion (%)')
        ax1.set_xlabel(category_column)
        
        for i, v in enumerate(summary_df['Bad_Proportion']):
            ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    if plot_type in ['both', 'count']:
        plt.figure(figsize=figsize)
        
        df_plot = pd.DataFrame({
            'Good': total_counts - bad_counts,
            'Bad': bad_counts
        })
        
        df_plot.plot(kind='bar', stacked=True, 
                    color=['forestgreen', 'darkred'])
        
        plt.title(f'Customer Distribution by {category_column}')
        plt.xlabel(category_column)
        plt.ylabel('Count')
        
        # for i, total in enumerate(total_counts):
        #     plt.text(i, total/2, f'Total: {total}', 
        #             ha='center', va='center', color='white', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Customer Type')
        plt.tight_layout()
        plt.show()
    
    print("\nSummary Statistics:")
    print("-" * 50)
    print(summary_df.to_string())
    
    # Print risk analysis
    print("\nRisk Analysis:")
    print("-" * 50)
    avg_bad_rate = (dataset[target_column].mean() * 100)
    print(f"Average Bad Rate: {avg_bad_rate:.2f}%")
    
    high_risk_categories = summary_df[summary_df['Bad_Proportion'] > avg_bad_rate]
    if not high_risk_categories.empty:
        print("\nHigh Risk Categories (Above Average):")
        for idx, row in high_risk_categories.iterrows():
            print(f"{idx}: {row['Bad_Proportion']:.2f}% "
                  f"({row['Bad_Count']} out of {row['Total_Count']} customers)")
            

def create_num_groups(
        df: pd.DataFrame, 
        num_col:str,
        bins : List[int]
):
    # Create age group ranges
    bins = bins #list(range(20, 71, 5))  # Creates bins: 20-25, 26-30, ..., 65-70
    labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
    
    # Add age group column
    df[f'{num_col}_binned'] = pd.cut(df[num_col], bins=bins, labels=labels, right=False)
    
    # Calculate metrics for each age group
    num_col_analysis = df.groupby(f'{num_col}_binned').agg({
        'TARGET': ['count', 'mean'],  # count = total customers, mean = default rate
        num_col: 'count'
    }).round(4)
    
    # Flatten column names
    num_col_analysis.columns = ['total_customers', 'default_rate', 'total_count']
    
    # Calculate percentage of total customers
    total_customers = num_col_analysis['total_customers'].sum()
    num_col_analysis['customer_percentage'] = (num_col_analysis['total_customers'] / total_customers * 100).round(2)
    
    return num_col_analysis

def plot_age_group_analysis(num_col_analysis : pd.DataFrame
                            ,num_col: str):
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Default Rate by Age Group
    sns.barplot(x=num_col_analysis.index, 
                y=num_col_analysis['default_rate'] * 100,
                ax=ax1,
                color='skyblue')
    ax1.set_title(f'Default Rate by {num_col} Group')
    ax1.set_xlabel('Group')
    ax1.set_ylabel(f'Default Rate ({num_col}%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(num_col_analysis['default_rate']):
        ax1.text(i, v*100, f'{(v*100):.1f}%', ha='center', va='bottom')
    
    # Plot 2: Customer Distribution Across Age Groups
    sns.barplot(x=num_col_analysis.index,
                y=num_col_analysis['customer_percentage'],
                ax=ax2,
                color='lightgreen')
    ax2.set_title(f'Customer Distribution Across {num_col} Groups')
    ax2.set_xlabel(f'{num_col} Group')
    ax2.set_ylabel('Percentage of Total Customers')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(num_col_analysis['customer_percentage']):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def print_age_group_insights(num_col_analysis):
    # Find the group with highest default rate
    highest_default = num_col_analysis['default_rate'].idxmax()
    # Find the group with lowest default rate
    lowest_default = num_col_analysis['default_rate'].idxmin()
    # Find the largest age group
    largest_group = num_col_analysis['customer_percentage'].idxmax()
    
    print("Key Insights from Age Group Analysis:")
    print("-" * 50)
    print(f"Highest Default Rate: {num_col_analysis.loc[highest_default, 'default_rate']*100:.1f}% (Age Group: {highest_default})")
    print(f"Lowest Default Rate: {num_col_analysis.loc[lowest_default, 'default_rate']*100:.1f}% (Age Group: {lowest_default})")
    print(f"Largest Age Group: {largest_group} ({num_col_analysis.loc[largest_group, 'customer_percentage']:.1f}% of customers)")
    print("\nDetailed Analysis:")
    print(num_col_analysis.to_string(float_format=lambda x: '{:.2f}'.format(x)))


def process_bureau_data(bureau_df, bureau_balance_df):
    """
    Process bureau.csv and bureau_balance.csv to get credit bureau history
    """
    # Process bureau_balance status
    # status_mapping = {
    #     'C': 'Closed',
    #     'X': 'Unknown',
    #     '0': 'No DPD',
    #     '1': 'DPD 1-30',
    #     '2': 'DPD 31-60',
    #     '3': 'DPD 61-90',
    #     '4': 'DPD 91-120',
    #     '5': 'DPD 120+'
    # }
    
    # # Aggregate bureau_balance by loan
    # balance_agg = bureau_balance_df.groupby('SK_ID_BUREAU').agg({
    #     'STATUS': lambda x: pd.Series.mode(x)[0],  # Most common status
    #     'MONTHS_BALANCE': ['count', 'min', 'max', 
    #                     lambda x: x.std(ddof=0) if len(x) > 1 else 0  ]  # History length
    # })

    # balance_agg.columns = ['SK_ID_BUREAU',
    #  'STATUS',
    #  'MONTHS_BALANCE_count',
    #  'MONTHS_BALANCE_min',
    #  'MONTHS_BALANCE_max'     ]
    
    # Aggregate bureau data
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',  # Number of previous credits
        'CREDIT_ACTIVE': lambda x: (x == 'Active').mean(),  # Ratio of active credits
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max']
    })
    
    # Rename columns for clarity
    bureau_agg.columns = [
        'n_credits',
        'active_credit_ratio',
        'avg_days_overdue',
        'max_days_overdue',
        'avg_max_overdue_amt',
        'max_max_overdue_amt',
        'avg_current_overdue',
        'max_current_overdue'
    ]
    
    return bureau_agg

def process_previous_applications(prev_app_df, pos_cash_df, credit_card_df, installments_df):
    """
    Process previous application data and related payment histories
    """
    # Process POS/Cash loans
    pos_agg = pos_cash_df.groupby('SK_ID_PREV').agg({
        'SK_DPD': ['mean', 'max'],
        'SK_DPD_DEF': ['mean', 'max']
    }).reset_index()
    
    # Flatten column names for pos_agg
    pos_agg.columns = ['SK_ID_PREV'] + [f'pos_{c[0]}_{c[1]}' for c in pos_agg.columns[1:]]
    
    # Process credit card data
    cc_agg = credit_card_df.groupby('SK_ID_PREV').agg({
        'SK_DPD': ['mean', 'max'],
        'SK_DPD_DEF': ['mean', 'max']
    }).reset_index()
    
    # Flatten column names for cc_agg
    cc_agg.columns = ['SK_ID_PREV'] + [f'cc_{c[0]}_{c[1]}' for c in cc_agg.columns[1:]]
    
    # Process installments
    inst_agg = installments_df.groupby('SK_ID_PREV').agg({
        'DAYS_ENTRY_PAYMENT': 'mean',
        'DAYS_INSTALMENT': 'mean',
        'AMT_PAYMENT': 'sum',
        'AMT_INSTALMENT': 'sum'
    }).reset_index()
    
    # Calculate payment metrics
    inst_agg['payment_delay'] = abs(inst_agg['DAYS_ENTRY_PAYMENT'] - inst_agg['DAYS_INSTALMENT'])
    inst_agg['payment_ratio'] = abs(inst_agg['AMT_PAYMENT'] / inst_agg['AMT_INSTALMENT'])
    
    
    # Ensure prev_app_df has a proper index
    prev_app_df = prev_app_df.reset_index(drop=True)
    
    # Merge all previous loan data
    prev_loan_metrics = prev_app_df.merge(pos_agg, on='SK_ID_PREV', how='left')
    prev_loan_metrics = prev_loan_metrics.merge(cc_agg, on='SK_ID_PREV', how='left')
    prev_loan_metrics = prev_loan_metrics.merge(inst_agg, on='SK_ID_PREV', how='left')
    
    # Aggregate to client level
    client_prev_loans = prev_loan_metrics.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',  # Number of previous applications
        'payment_delay': ['mean', 'max'],
        'payment_ratio': 'mean',
        'pos_SK_DPD_mean': 'mean',
        'pos_SK_DPD_max': 'max',
        'pos_SK_DPD_DEF_mean': 'mean',
        'pos_SK_DPD_DEF_max': 'max',
        'cc_SK_DPD_mean': 'mean',
        'cc_SK_DPD_max': 'max',
        'cc_SK_DPD_DEF_mean': 'mean',
        'cc_SK_DPD_DEF_max': 'max'
    })
    
    # Flatten the column names in the final result
    client_prev_loans.columns = [f'{c[0]}_{c[1]}' if isinstance(c, tuple) else c 
                                for c in client_prev_loans.columns]
    
    return client_prev_loans

def merge_all_credit_data(application_df, bureau_agg, client_prev_loans):
    """
    Merge all credit history data with current application
    """
    # Merge bureau data
    df = application_df.join(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Merge previous loan data
    df = df.join(client_prev_loans, on='SK_ID_CURR', how='left')
    
    # Fill missing values for clients with no history
    df = df.fillna({
        'n_credits': 0,
        'active_credit_ratio': 0,
        'avg_days_overdue': 0,
        'max_days_overdue': 0,
        'payment_delay': 0,
        'payment_ratio': 1
    })
    
    return df

def calculate_risk_score(row):
    """
    Calculate risk score based on credit history with simplified error handling
    """
    try:
        # Define default values for missing fields
        max_days_overdue = row.get('max_days_overdue', 0)
        max_overdue_amt = row.get('max_max_overdue_amt', 0)
        payment_delay = row.get('payment_delay_max', 0)
        payment_ratio = row.get('payment_ratio', 1)  # Default to 1 if missing
        dpd_max = row.get('SK_DPD_max', 0)
        
        # Count risk factors
        risk_count = 0
        if max_days_overdue > 0: risk_count += 1
        if max_days_overdue > 60: risk_count += 1
        if max_overdue_amt > 0: risk_count += 1
        if payment_delay > 0: risk_count += 1
        if payment_ratio < 0.95: risk_count += 1
        if dpd_max > 60: risk_count += 1
        
        # Simple risk score: proportion of risk factors present
        return risk_count / 6
    except:
        return 0  # Return 0 if any error occurs

def get_segment_performance(df, segment_column):
    """
    Calculate performance metrics for each segment with simplified error handling
    """
    segment_metrics = []
    
    for segment in df[segment_column].unique():
        try:
            segment_data = df[df[segment_column] == segment]
            
            # Calculate basic metrics
            metrics = {
                'segment': segment,
                'n_loans': len(segment_data),
                'default_rate': segment_data['TARGET'].mean(),
                'avg_risk_score': segment_data.apply(calculate_risk_score, axis=1).mean(),
            }
            
            # Add additional metrics if columns exist
            additional_metrics = {
                'active_credit_ratio': 'active_credit_ratio',
                'avg_days_overdue': 'days_overdue_mean', 
                'max_days_overdue': 'days_overdue_max',
                'payment_delay': 'payment_delay_max',
                'payment_ratio': 'payment_ratio'
            }
            
            # Calculate average for each metric if column exists
            for metric_name, column_name in additional_metrics.items():
                if column_name in segment_data.columns:
                    metrics[metric_name] = segment_data[column_name].fillna(0).mean()
                else:
                    metrics[metric_name] = None  # or 0 depending on your preference
            
            # Simple decision rules
            if metrics['default_rate'] > 0.15:
                metrics['decision'] = 'RESTRICT'
            elif metrics['default_rate'] < 0.05:
                metrics['decision'] = 'EXPAND'
            else:
                metrics['decision'] = 'INVESTIGATE'
                
            segment_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing segment {segment}: {str(e)}")
            # If there's any error, add basic info only
            segment_metrics.append({
                'segment': segment,
                'n_loans': len(df[df[segment_column] == segment]),
                'default_rate': None,
                'avg_risk_score': None,
                'active_credit_ratio': None,
                'avg_days_overdue': None,
                'max_days_overdue': None,
                'payment_delay': None,
                'payment_ratio': None,
                'decision': 'INVESTIGATE'
            })
    
    return pd.DataFrame(segment_metrics)


import pandas as pd
import numpy as np

def calculate_cross_segment_metrics(df, feature1, feature2, 
                                  target_col='TARGET',
                                  credit_amount_col='AMT_CREDIT',
                                  payment_delay_col='PAYMENT_DELAY',
                                  min_samples=50):
    """
    Calculate metrics for cross-segments of two features
    
    Parameters:
    df: DataFrame with the loan data
    feature1: First feature name for segmentation
    feature2: Second feature name for segmentation
    target_col: Target column name (default)
    credit_amount_col: Credit amount column name
    payment_delay_col: Payment delay column name
    min_samples: Minimum number of samples required for a segment
    
    Returns:
    Tuple of (DataFrame with metrics, Pivot tables for key metrics)
    """
    
    # Create cross-segments
    df['segment'] = df[feature1].astype(str) + ' x ' + df[feature2].astype(str)
    
    # Calculate base metrics
    segment_metrics = []
    
    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        n_loans = len(segment_data)
        
        # Skip segments with too few samples
        if n_loans < min_samples:
            continue
            
        # Extract individual features for the segment
        feat1_val, feat2_val = segment.split(' x ')
        
        total_credit = segment_data[credit_amount_col].sum()
        default_rate = segment_data[target_col].mean()
        total_loss = total_credit * default_rate
        
        metrics = {
            'segment': segment,
            feature1: feat1_val,
            feature2: feat2_val,
            'n_loans': n_loans,
            'default_rate': default_rate,
            'total_credit': total_credit,
            'avg_credit': total_credit/n_loans,
            'total_loss': total_loss,
            'avg_loss': total_loss/n_loans
        }
        
        if payment_delay_col in segment_data.columns:
            metrics['avg_payment_delay'] = segment_data[payment_delay_col].mean()
            
        segment_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(segment_metrics)
    
    # Create pivot tables for key metrics
    pivot_tables = {
        'default_rate': pd.pivot_table(
            metrics_df, 
            values='default_rate', 
            index=feature1,
            columns=feature2,
            aggfunc='mean'
        ).round(4),
        
        'avg_credit': pd.pivot_table(
            metrics_df,
            values='avg_credit',
            index=feature1,
            columns=feature2,
            aggfunc='mean'
        ).round(2),
        
        'n_loans': pd.pivot_table(
            metrics_df,
            values='n_loans',
            index=feature1,
            columns=feature2,
            aggfunc='sum'
        ),
        
        'total_loss': pd.pivot_table(
            metrics_df,
            values='total_loss',
            index=feature1,
            columns=feature2,
            aggfunc='sum'
        ).fillna(0).round(2)
    }
    
    # Format metrics for display
    display_metrics = metrics_df.copy()
    display_metrics['n_loans'] = display_metrics['n_loans'].apply(lambda x: f"{x:,}")
    display_metrics['default_rate'] = display_metrics['default_rate'].apply(lambda x: f"{x:.2%}")
    display_metrics['total_credit'] = display_metrics['total_credit'].apply(lambda x: f"{x:,.2f}")
    display_metrics['avg_credit'] = display_metrics['avg_credit'].apply(lambda x: f"{x:,.2f}")
    display_metrics['total_loss'] = display_metrics['total_loss'].apply(lambda x: f"{x:,.2f}")
    display_metrics['avg_loss'] = display_metrics['avg_loss'].apply(lambda x: f"{x:,.2f}")
    
    if 'avg_payment_delay' in display_metrics.columns:
        display_metrics['avg_payment_delay'] = display_metrics['avg_payment_delay'].apply(lambda x: f"{x:.2f}")
    
    return display_metrics, pivot_tables

def plot_cross_segment_heatmap(pivot_table, title, fmt='.2%', figsize=(12, 8), is_currency=False):
  
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    if is_currency:
        # For currency values, use a custom formatter
        def currency_formatter(x, p):
            if x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1e3:
                return f'${x/1e3:.1f}K'
            else:
                return f'${x:.0f}'
        
        # Create heatmap with custom annotation format
        sns.heatmap(pivot_table,
                   annot=True,
                   fmt='.0f',
                   cmap='YlOrRd',
                   center=pivot_table.mean().mean(),
                   annot_kws={'size': 8},
                   cbar_kws={'format': currency_formatter})
    else:
        # Default heatmap for percentages or other metrics
        sns.heatmap(pivot_table,
                   annot=True,
                   fmt=fmt,
                   cmap='YlOrRd',
                   center=pivot_table.mean().mean())
    
    plt.title(title)
    plt.tight_layout()
    return plt


def calculate_segment_metrics(df, segment_column, 
                            target_col='TARGET',
                            credit_amount_col='AMT_CREDIT',
                            payment_delay_col='PAYMENT_DELAY'):
    segment_metrics = []
    
    for segment in df[segment_column].unique():
        try:
            # Get segment data
            segment_data = df[df[segment_column] == segment]
            
            # Calculate metrics
            n_loans = len(segment_data)
            total_credit = segment_data[credit_amount_col].sum()
            default_rate = segment_data[target_col].mean()
            total_loss = total_credit * default_rate
            
            metrics = {
                'segment': segment,
                'n_loans': f"{n_loans:,}",
                'default_rate': f"{default_rate:.6f}",
                'payment_delay': (f"{segment_data[payment_delay_col].mean():.6f}" 
                                if payment_delay_col in segment_data.columns else None),
                'loss': f"{total_loss:,.2f}",
                'avg_loss': f"{(total_loss/n_loans):,.2f}",  # Added average loss per loan
                'total_credit': f"{total_credit:,.2f}",
                'avg_credit': f"{(total_credit/n_loans):,.2f}"
            }
            
            segment_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error processing segment {segment}: {str(e)}")
            segment_metrics.append({
                'segment': segment,
                'n_loans': '0',
                'default_rate': None,
                'payment_delay': None,
                'loss': None,
                'avg_loss': None,
                'total_credit': None,
                'avg_credit': None
            })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(segment_metrics)
    
    # Sort by number of loans descending (converting back to numeric for sorting)
    metrics_df['n_loans_sort'] = pd.to_numeric(metrics_df['n_loans'].str.replace(',', ''))
    metrics_df = metrics_df.sort_values('n_loans_sort', ascending=False)
    metrics_df = metrics_df.drop('n_loans_sort', axis=1)
    
    return metrics_df
