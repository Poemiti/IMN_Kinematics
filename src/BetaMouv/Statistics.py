# src/Base/Statistics.py

from pathlib import Path
import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.multitest import multipletests


class Statistics:
    """Class to compute statistic on scalar metrics computed for several trials
    Dataframe must contain the following columns : 
    trial_name, metric1, metric2, metric3, laser_type, laser_state, laser_intensity,"""

    def __init__(self, data: pd.DataFrame, metric: str):

        self.metric = metric

        self.data = data
        self.data["laser_group"] = data["laser_cond"] + "." + data["laser_intensity"]
        


    def kruskal_test(self, group):

        samples = [
            sub_group[self.metric].values
            for _, sub_group in group.groupby("condition")
        ]
        
        # need at least 2 groups
        if len(samples) < 2:
            return pd.Series({
                "n_groups": len(samples),
                "kruskal_stat": None,
                "p_value": None
            })
        
        stat, p = stats.kruskal(*samples, nan_policy="omit")
        
        return pd.Series({
            "n_groups": len(group),
            "kruskal_stat": stat,
            "p_value": p
        })

    def shapiro_test(self, group):
        stat, p = stats.shapiro(group[self.metric])
        return pd.Series({
            "n": len(group),
            "shapiro_stat": stat,
            "p_value": p
        })

    def mann_whitney(self, comparisons: list, data: pd.DataFrame = None) : 
        if data is None: 
            data = self.data

        results = []

        for g1, g2 in comparisons:
            x = data.loc[data["laser_group"] == g1, self.metric]
            y = data.loc[data["laser_group"] == g2, self.metric]

            # print(g1, len(x), np.isnan(x).sum())
            # print(g2, len(y), np.isnan(y).sum())
            
            if len(x) > 0 and len(y) > 0:
                stat, p = stats.mannwhitneyu(x, y, alternative="two-sided", nan_policy="omit")
            else:
                stat, p = None, None
            
            results.append({
                "group1": g1,
                "group2": g2,
                "n1": len(x),
                "n2": len(y),
                "U_stat": stat,
                "p_value": p
            })


        pairwise_results = pd.DataFrame(results)
        print(pairwise_results)
        pairwise_results["p_adj"] = multipletests(pairwise_results["p_value"], method="bonferroni")[1]
        
        return pairwise_results


    def compute_statistics(self, comparisons: list[tuple[str, str]], data: pd.DataFrame = None) : 
        """COmpute the following test : 
        - shapiro (normality check)
        - kruskal (non parametric variance analysis)
        - if kruskal significant : pairwise Mann Whitney + Bofferoni correction"""
        
        if data is None: 
            data = self.data

        stat_res = {}

        # 1. check normality
        data_normality = (data
                            .groupby(["laser_cond", "laser_intensity"])
                            .apply(self.shapiro_test, include_groups=False)
                            .reset_index()
                        )
        print("\nShapiro :\n", data_normality)

        stat_res["shapiro"] = data_normality

        # 2. Kruskal test
        data_kruskal = self.kruskal_test(data)
        print("\nKruskal :\n", data_kruskal)

        
        stat_res["kruskal"] = data_kruskal

        if data_kruskal["p_value"] > 0.05 : 
            print(f"\nKrustal wallis test is NOT significant: ")
            return stat_res

        pairwise_results = self.mann_whitney(comparisons)
        stat_res["mann_whitney"] = pairwise_results

        print("\nMann Whitney :\n", pairwise_results)

        return stat_res




    ############ permutation ##################


    def permutation_test(self, group1, group2, n_perm=10000):
        values1 = group1[self.metric].to_numpy()
        values2 = group2[self.metric].to_numpy()

        n1 = len(values1)
        n2 = len(values2)
        print(n1, n2)
        combined = np.concatenate([values1, values2])
        
        # observed difference
        observed_diff = values1.mean() - values2.mean()
        
        perm_diff = np.zeros(n_perm)
        for i in range(n_perm):
            shuffled = np.random.permutation(combined)
            perm_diff[i] = shuffled[:n1].mean() - shuffled[n1:n1+n2].mean()
        
        # two-tailed p-value
        p_value = np.mean(perm_diff <= observed_diff)
        
        return observed_diff, perm_diff, p_value


    def compute_permutation_effect_size(self, data: pd.DataFrame, n_perm: int) -> dict : 
        if data is None:
            data = self.data

        beta_on = data[
            (data["laser_type"] == "Beta") &
            (data["laser_state"] == "LaserOn")
        ]

        conti_on = data[
            (data["laser_type"] == "Conti") &
            (data["laser_state"] == "LaserOn")
        ]

        off = data[data["laser_state"] == "LaserOff"]

        # beta_on = data[
        #     (data["condition"] == "Beta") &
        #     (data["laser_state"] == "LaserOn")
        # ]
        # beta_off = data[
        #     (data["condition"] == "Beta") &
        #     (data["laser_state"] == "LaserOff")
        # ]

        # conti_on = data[
        #     (data["condition"] == "Conti") &
        #     (data["laser_state"] == "LaserOn")
        # ]
        # conti_off = data[
        #     (data["condition"] == "Conti") &
        #     (data["laser_state"] == "LaserOff")
        # ]


        # Permutation test 
        b_observed_diff, b_perm_diff, b_pval = self.permutation_test(beta_on, off, n_perm)
        c_observed_diff, c_perm_diff, c_pval = self.permutation_test(conti_on, off, n_perm)
        bc_observed_diff, bc_perm_diff, bc_pval = self.permutation_test(conti_on, beta_on, n_perm)

        print("Beta vs OFF", b_pval)
        print("Conti vs OFF", c_pval)
        print("Beta vs Conti", bc_pval)

        results = [
            {
                "Condition" : "Beta vs OFF",
                "observed mean difference" : b_observed_diff,
                "permutation differences" : b_perm_diff,
                "p_value" : b_pval,
            },
            {
                "Condition" : "Conti vs OFF",
                "observed mean difference" : c_observed_diff,
                "permutation differences" : c_perm_diff,
                "p_value" : c_pval,
            },
            {
                "Condition" : "Beta vs Conti",
                "observed mean difference" : bc_observed_diff,
                "permutation differences" : bc_perm_diff,
                "p_value" : bc_pval,
            }
        ]

        res = pd.DataFrame(results)
        df = res.set_index("Condition")
        print(df[["observed mean difference", "p_value"]].T)

        return results