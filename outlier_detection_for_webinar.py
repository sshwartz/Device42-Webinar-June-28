from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv
from pyod.models.copod import COPOD
import seaborn as sns
from sklearn.ensemble import IsolationForest

class outlier (object):
    def __init__(self):
        self.devices_dict = {}
        self.services_dict = {}
        self.dev_svce_dict = {}
        self.dev_os = {}
        print("\n")

    def run_copod(self):
        X = self.preprocess_services_data()
        clf_name = 'COPOD'
        clf = COPOD()
        clf.fit(X)
        scores = clf.decision_function(X)
        n = 0
        for score in scores:
            if score > 700:
                self.print_device_info(n, score)
            n += 1

        sns.displot(data=scores)
        plt.legend()
        plt.xlabel("Outlier score")
        plt.show()

    def run_isolation_forest(self):
        X = self.preprocess_services_data()
        clf = IsolationForest(random_state=0).fit(X)
        clf.fit(X)
        scores = clf.decision_function(X)
        n = 0
        for score in scores:
            if score < .15:
                self.print_device_info(n, score)
            n += 1

        sns.displot(data=scores)
        plt.legend()
        plt.xlabel("Outlier score")
        plt.show()

    def run_local_outlier_factor(self):
        X = self.preprocess_services_data()
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

        # No decision_function, fit, or score_samples
        y_pred = clf.fit_predict(X)
        X_scores = clf.negative_outlier_factor_
        n = 0
        for score in X_scores:
            if score < -1:
                self.print_device_info(n, score)
            n += 1

        sns.displot(data=X_scores)
        plt.legend()
        plt.xlabel("Outlier score")
        plt.show()

    def preprocess_services_data(self):
        with open("/home/data/device_service.txt", 'r') as file_obj:
            reader = csv.reader(file_obj)
            line_no = 1
            dev_no = 0
            svce_no = 0
            try:
                for row in reader:
                    line_no += 1
                    device = row[0]
                    service = row[1]
                    try:
                        current_dev_no = self.devices_dict[device]
                    except:
                        current_dev_no = dev_no
                        dev_no += 1
                    self.devices_dict[device] = current_dev_no
                    try:
                        current_svce_no = self.services_dict[service]
                    except:
                        current_svce_no = svce_no
                        svce_no += 1
                    self.services_dict[service] = current_svce_no
                    try:
                        services = self.dev_svce_dict[current_dev_no]
                    except:
                        self.dev_svce_dict[current_dev_no] = []
                    self.dev_svce_dict[current_dev_no].append(current_svce_no)
            except Exception as e:
                print("error in line %d: %s %s" % (line_no, str(type(e)), str(e)))
            X = np.zeros((dev_no, svce_no))
            for dev_no in self.dev_svce_dict:
                for svce_no in self.dev_svce_dict[dev_no]:
                    X[dev_no, svce_no] = 1
            return X

    def print_device_info (self, n, score):
        for device in self.devices_dict:
            if self.devices_dict[device] == n:
                dev_no = self.get_dev_no(device)
                print("Device name: %s.  Score: %d." % (device, score))
                print(end="Services: ")
                nchar = 0
                for svce_no in self.dev_svce_dict[n]:
                    svce_name = self.get_svce_name(svce_no)
                    nchar += len(svce_name) + 2
                    if nchar < 215:
                        print(end = "%s, " % svce_name)
                    else:
                        print(end="%s\n" % svce_name)
                        nchar = len(svce_name) + 2
                print("\n")

    def get_os_name(self, device):
        for d in self.dev_os:
            if d == device:
                return(self.dev_os[d])

    def get_dev_no (self, device):
        for dev_name in self.devices_dict:
            if dev_name == device:
                return self.devices_dict[device]

    def get_dev_name(self, dev_no):
        for device in self.devices_dict:
            if self.devices_dict[device] == dev_no:
                return device

    def get_svce_name(self, svce_no):
        for svce_name in self.services_dict:
            if self.services_dict[svce_name] == svce_no:
                return svce_name

out = outlier()
out.run_copod()
# out.run_isolation_forest()
# out.run_local_outlier_factor()
