
# imports

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms, datasets
from PIL import Image
import torch.nn as nn
import torch 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt


class init_dataset_views(Dataset):
    # rootdir: mandatory1_data
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_names = []
        self.labels = []
        self.transform = transform

        image_names_all, labels_all = [], []

        label_nr = {"buildings":0, "forest":1, "glacier":2, "mountain":3, "sea":4, "street":5}

        # loading all image paths and labels for all classes into arays
        for dir in os.listdir(self.root_dir):
            path_to_img = os.path.join(self.root_dir, dir)
            if not os.path.isfile(path_to_img):
                for file in os.listdir(path_to_img):
                    if file.endswith(".jpg"):
                        image_name = os.path.join(path_to_img, file)
                        image_names_all.append(image_name)
                        labels_all.append(label_nr[dir]) # the label of the image is what the directory it is in is called

            else: # then it is a file
                if path_to_img.endswith(".JPEG"):
                    image_names_all.append(path_to_img)
                    labels_all.append("") # in this case we dont need the labels, but we need a filed labels list

                
        # splitting to get only test-data (vs. all the other data)
        test_size = 3000/len(image_names_all)
        X_all, X_test, y_all, y_test = train_test_split(image_names_all, labels_all, test_size=test_size, stratify=labels_all, random_state=0)
        
        # splitting the rest of the data into training and validation data
        val_size = 2000/len(X_all)
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=val_size, stratify=y_all, random_state=0)
        
        self.datasets = [X_train, y_train, X_val, y_val, X_test, y_test]

        self.verify_disjointness(X_train, X_val, X_test)


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        filename = self.image_names[index]
        image = Image.open(filename).convert("RGB")
        label = self.labels[index]

        if self.transform: # if a transform is given
            image = self.transform(image)
        else: 
            image = transforms.ToTensor()(image)

        sample = {"image": image, "label": label, "filename": filename}

        return sample
    
    
    def set_trvaltest(self, dataset_split):
        if dataset_split == 0: # training data
            self.image_names = self.datasets[0]
            self.labels = self.datasets[1]

        elif dataset_split == 1: # validation data
            self.image_names = self.datasets[2]
            self.labels = self.datasets[3]

        elif dataset_split == 2: # test data
            self.image_names = self.datasets[4]
            self.labels = self.datasets[5]


    def verify_disjointness(self, training, validation, test):
        tr_set, va_set, te_set = set(training), set(validation), set(test)

        train_val_check = tr_set.isdisjoint(va_set) # checking training set vs validation set
        train_test_check = tr_set.isdisjoint(te_set) # checking training set vs test set
        val_test_check = va_set.isdisjoint(te_set) # checking validation set vs test set

        if train_val_check and train_test_check and val_test_check:
            print("All datasets are disjoint :)")
        else:
            print("The datasets are not disjoint :( Please fix it")




def dataloaders(root_dir, batchsize_training, batchsize_test, transform=None):
    # creating the dataset
    datasets_views = init_dataset_views(root_dir=root_dir, transform=transform)

    datasets = {}
    datasets_views.set_trvaltest(dataset_split=0) # getting the training data
    datasets["training"] = datasets_views

    datasets_views.set_trvaltest(dataset_split=1) # getting the validation data
    datasets["validation"] = datasets_views

    datasets_views.set_trvaltest(dataset_split=2) # getting the test data
    datasets["test"] = datasets_views

    # loading the data using dataloaders
    dataloaders = {}
    dataloaders["training"] = DataLoader(datasets["training"], batch_size=batchsize_training, shuffle=True)
    dataloaders["validation"] = DataLoader(datasets["validation"], batch_size=batchsize_test, shuffle=False)
    dataloaders["test"] = DataLoader(datasets["test"], batch_size=batchsize_test, shuffle=False)

    return dataloaders


def transformation(resize=224, crop=224, scale1=[0.485, 0.456, 0.406], scale2=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(scale1, scale2)
    ])

    return transform


def class_accuracy(true, pred):
    confusion = confusion_matrix(true, pred)
 

    sum_true_class = sum(np.transpose(confusion))
    nr_classes = confusion.shape[0]

    all_accuracies = []
  
    for i in range(nr_classes):
        accuracy = confusion[i][i]/sum_true_class[i]
        all_accuracies.append(accuracy)
        #print(f"Accuracy for class {i+1}: {accuracy}")

    accuracy_mean = np.mean(all_accuracies)
    #print(f"Mean accuracy over all classes: {np.mean(all_accuracies)}\n")
    

    return all_accuracies, accuracy_mean



def average_precision(true, pred):

    nr_classes = np.max(true)+1
    ap_all = []
    true = np.asarray(true).reshape(-1, 1)
    pred = np.asarray(pred)

    onehot_enc = OneHotEncoder()
    onehot_enc.fit(true)
    onehot_labels_true = onehot_enc.transform(true).toarray()

    for i in range(nr_classes):
        # we use the columns of the true and pred lists because each column has values for the label and predicted
        # value of one class: column 1 - class 1, column 2 - class 2 and so on...

        ap_class = average_precision_score(onehot_labels_true[:,i], pred[:,i])
        ap_all.append(ap_class)

        #print(f"The average precision score for class {i+1}: {ap_class}")

    ap_mean = np.mean(ap_all)
    
    #print(f"The average precision score for all classes: {ap_mean}")

    return ap_all, ap_mean
    


def train_model_with_finetuning(model, device, dl_train, optimizer):

    model.to(device)

    CE_loss = nn.CrossEntropyLoss() # for calculating cross entropy loss
    # doing the training

    model.train() # setting the model in training mode

    tot_loss = []
    all_outputs, all_labels = [], []

    for i, data in enumerate(dl_train):
        img, label =  data["image"], data["label"]
        label = torch.as_tensor(label)
   
        img =  img.to(device)
        label = label.to(device)
    
        optimizer.zero_grad() # zeroing out previous gradients
        output = model(img) # calculating output from model
        loss = CE_loss(output, label) # calculating the loss based on output and labels
        
        loss.backward() # backpropagating the loss
        optimizer.step() # updating parameters
        tot_loss.append(loss.item())

        predictions = torch.max(output, 1)[1].to("cpu").numpy().tolist()
        labels = label.to("cpu").numpy().tolist()
       

        all_outputs = all_outputs + predictions 
        all_labels = all_labels + labels

    accuracies = class_accuracy(all_labels, all_outputs)

    return np.mean(tot_loss), accuracies, model



def eval_model(model, dataloader, device):

    model.eval() # setting the model in evaluation mode

    all_losses = []
    all_labels, all_outputs = [], []
    all_labels_AP, all_outputs_AP = [], []


    CE_loss = nn.CrossEntropyLoss() # for calculating cross entropy loss

    with torch.no_grad(): # not calculating the gradients - we are not training the model
        for i, data in enumerate(dataloader):
            img = data["image"].to(device)
            label = data["label"]
            label = torch.as_tensor(label)
            label = label.to(device)

            outputs = model(img) # calculating the outputs of the model
            loss = CE_loss(outputs, label)
            all_losses.append(loss.item())

            predictions = torch.max(outputs, 1)[1].to("cpu").numpy().tolist()
            labels = label.to("cpu").numpy().tolist()

            all_outputs = all_outputs + predictions
            all_labels = all_labels + labels
            all_labels_AP += label.to("cpu").numpy().tolist()
            all_outputs_AP += outputs.to("cpu").numpy().tolist()

    accuracies = class_accuracy(all_labels, all_outputs)
    aps = average_precision(all_labels_AP, all_outputs_AP)

    
    return np.mean(all_losses), accuracies, aps




def print_stuff(accuracy, ap, model_nr):
    
    print(f"\nStatistics for model {model_nr}:\n")

    all_accuracies = accuracy[0]
    mean_accuracy = accuracy[1]
    all_ap = ap[0]
    mean_ap = ap[1]

    for i in range(len(all_accuracies)):
        print(f"Accuracy for class {i+1}: {all_accuracies[i]}")

    print(f"Mean accuracy over all classes: {mean_accuracy}\n")

    for i in range(len(all_ap)):
        print(f"The average precision score for class {i+1}: {all_ap[i]}")

    print(f"The average precision score for all classes: {mean_ap}")    
       


def train_model(device, model, config, dl, optimizer):    
    all_losses_train, all_losses_val = [], []
    mean_accuracy_train, mean_accuracy_val = [], []

    for epoch in range(config["epochs"]): # doing the training
        tot_loss_train, accuracies_train, model = train_model_with_finetuning(model, device, dl["training"], optimizer)
        tot_loss_val, accuracies_val, aps = eval_model(model, dl["validation"], device)

        all_losses_train.append(tot_loss_train)
        all_losses_val.append(tot_loss_val)

        mean_accuracy_train.append(accuracies_train[1])
        mean_accuracy_val.append(accuracies_val[1])
        
    # printing the accuracy per class and average precision per class for val set after all epochs are done
    print_stuff(accuracies_val, aps, config["model_nr"])

    return all_losses_train, all_losses_val, mean_accuracy_train, mean_accuracy_val, model



def plot_losses_accuracies(loss_train, loss_val, accuracy_train, accuracy_val):
    plt.figure(figsize=(18,16))
    
    fig1 = plt.subplot(2,1,1)
    fig1.plot(loss_train, label="Training loss")
    fig1.plot(loss_val, label="Validation loss")
    fig1.grid()
    fig1.legend(loc="upper right")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    fig2 = plt.subplot(2,1,2)
    fig2.plot(accuracy_train, label="Mean training accuracy")
    fig2.plot(accuracy_val, label="Mean validation accuracy")
    fig2.grid()
    fig2.legend(loc="lower right")
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")


def init_model(config):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # overriding the pretrained network's fully connected layer to match
    # the number of classes in our problem
    nr_features_base_model = model.fc.in_features
    model.fc = nn.Linear(nr_features_base_model, config["nr_classes"])

    return model


def train_models(root_dir, config, dl, device):

    accuracies_all_models = []
    model = init_model(config)
    lr = config["learning_rate"]

    # Trying out 3 models with different optimizers:
    config["model_nr"] = 1
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr)
    all_losses_train1, all_losses_val1, mean_accuracy_train1, mean_accuracy_val1, model1 = train_model(device, model, config, dl, optimizer1)
    accuracies_all_models.append(mean_accuracy_val1[-1]) # using only the last accuracy from the last epoch
    
    config["model_nr"] = 2
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=lr)
    all_losses_train2, all_losses_val2, mean_accuracy_train2, mean_accuracy_val2, model2 = train_model(device, model, config, dl, optimizer2)
    accuracies_all_models.append(mean_accuracy_val2[-1]) # using only the last accuracy from the last epoch

    config["model_nr"] = 3
    optimizer3 = torch.optim.SGD(model.parameters(), lr=lr)
    all_losses_train3, all_losses_val3, mean_accuracy_train3, mean_accuracy_val3, model3 = train_model(device, model, config, dl, optimizer3)
    accuracies_all_models.append(mean_accuracy_val3[-1]) # using only the last accuracy from the last epoch

    max_accuracy = np.argmax(accuracies_all_models)
    model = None
    if max_accuracy == 0:
        plot_losses_accuracies(all_losses_train1, all_losses_val1, mean_accuracy_train1, mean_accuracy_val1)
        model = model1

    elif max_accuracy == 1:
        plot_losses_accuracies(all_losses_train2, all_losses_val2, mean_accuracy_train2, mean_accuracy_val2)
        model = model2

    elif max_accuracy == 2:
        plot_losses_accuracies(all_losses_train3, all_losses_val3, mean_accuracy_train3, mean_accuracy_val3)
        model = model3

    return model



def test_model(model, dataloader, device):

    model.eval() # setting the model in evaluation mode

    all_losses = []
    all_labels, all_outputs = [], []
    all_labels_AP, all_outputs_AP = [], []

    softmax_scores = []
    predicitons_list = []

    images = []

    CE_loss = nn.CrossEntropyLoss() # for calculating cross entropy loss

    with torch.no_grad(): # not calculating the gradients - we are not training the model
        for i, data in enumerate(dataloader):
            img = data["image"].to(device)
            label = data["label"]
            label = torch.as_tensor(label)
            label = label.to(device)

            images += data["filename"]

            outputs = model(img).to(device) # calculating the outputs of the model      
            softmax_scores.append(outputs.detach().cpu())      
            
            predictions_tensor = torch.max(outputs, 1)[1].to("cpu")
            predicitons_list.append(predictions_tensor)

            loss = CE_loss(outputs, label)
            all_losses.append(loss.item())

            predictions = predictions_tensor.numpy().tolist()
            labels = label.to("cpu").numpy().tolist()

            all_outputs = all_outputs + predictions
            all_labels = all_labels + labels
            all_labels_AP += label.to("cpu").numpy().tolist()
            all_outputs_AP += outputs.to("cpu").numpy().tolist()

    accuracies = class_accuracy(all_labels, all_outputs)
    aps = average_precision(all_labels_AP, all_outputs_AP)

    return torch.cat(softmax_scores), torch.cat(predicitons_list), all_outputs_AP, images, accuracies, aps



def show_ranked_images(model, dl, device):
    softmax_scores, predictions, all_outputs_AP, images, accuracies, aps = test_model(model, dl["test"], device)

    to_display = [2, 3, 4]
    to_display_names = ["Glacier", "Mountains", "Sea"]


    all_images = {to_display[0]:[], to_display[1]:[], to_display[2]:[]}
    max_scores = {to_display[0]:[], to_display[1]:[], to_display[2]:[]}
    
    for out_list, img in zip(all_outputs_AP, images):
        max_out = max(out_list) # finding max score of class of an image 
        pred_class = out_list.index(max(out_list)) # finding index of max value - which is the predicted class

        if pred_class in to_display:
            max_scores[pred_class].append(max_out)
            all_images[pred_class].append(img)


    i = 0
    # sorting both softmax scores and images together in descending order
    for img_key, score_key in zip(all_images, max_scores):
        sorted_lists = sorted(zip(max_scores[score_key], all_images[img_key]))[::-1]

        sorted_scores, sorted_images = zip(*sorted_lists) # unzipping images

        top_10 = sorted_images[:10] # choosing the top ten images
        bottom_10 = sorted_images[::-1][:10] # choosing the bottom ten images

        
        fig, axes = plt.subplots(1, 10, figsize=(18, 3))
        plt.title(f"Top 10 ranked for class: {to_display_names[i]}")
        for img, ax in zip(top_10, axes):
            image = Image.open(img).convert("RGB")
            ax.imshow(image)
            ax.axis("off")

        fig, axes = plt.subplots(1, 10, figsize=(18, 3))
        plt.title(f"Bottom 10 ranked for class: {to_display_names[i]}")
        for img, ax in zip(bottom_10, axes):
            image = Image.open(img).convert("RGB")
            ax.imshow(image)
            ax.axis("off")

        i += 1
    
        plt.show()
        


percentage = {0:0, 1:0, 2:0, 3:0, 4:0}

def hook_stats1(module, input, output):

    # called every time a forward pass is made through a module
    # input er input til modulen (bilde)
    # output er output fra modulen (feature map)
    # output[0] = 5 tensorer av 1 bilde gjennom 5 convolutional filters

    
    for i in range(output.shape[0]): # iterating through the output-tensor
        out = output[i].detach().cpu()
        mask = torch.where(out <= 0, torch.tensor(1), torch.tensor(0)) # setting all negative values/0 to 1 and positive values to 0
        non_positive_mean = mask.mean(dtype=float) # calculating the mean of non-positive values for one image

        percentage[(hook_stats1.nr)%5] += non_positive_mean.numpy()

    hook_stats1.nr += 1


def get_feature_maps(hook, dataloader, model, device, config):
    modules = {}
    hook.nr = 0
    # this will add hooks to the convolution outputs that we want - so we get outputs for 5 feature maps
    for name, module in model.named_modules():
        if "conv1" == name or "1.conv1" in name: # choosing some of the modules 
            this = module.register_forward_hook(hook) # hooking these modules
            modules[name] = this

    # we go through around 200 images and calculating the output for them
    count = 0
    for i, data in enumerate(dataloader):

        if isinstance(data, list): # if its a list, then its the cifar (in this case) dataset
            img = data[0].to(device)
        else:
            img = data["image"].to(device)
            
        count += config["batchsize_test"]
        # we only want statistics for 200 images - when we are over 200, we want to break and only keep the amount
        # of images up to 200 and discard the rest
        if count > 200:
            img = img[0:(count-200)] # trimming the last tensor to have exactly 200 images
            out = model(img).to(device)
            break

        # when going through the model and we get to one of the models that are hooked, 
        # the callback method will be called
        out = model(img).to(device)

    # removing the handle
    for module in modules.keys():
        modules[module].remove()

    return count, modules


def feature_map_statistics(dataloader, config, model, device):
    count, modules = get_feature_maps(hook_stats1, dataloader["validation"], model, device, config)
    
    result_statistics = {}
    for i, j in zip(percentage.keys(), modules):
        res = percentage[i]/count * 100
        result_statistics[j] = res

    return result_statistics



def print_feature_map_statistics(stats_dict):
    print("Feature map statistics part 1:\n")
    print("Convolutional layer - percentage")
    for key in stats_dict.keys():
        print(f"{key} - {stats_dict[key]:.1f}")


# defining a hook for the second statistics task, where we calculate other measures
mean_feature_map = {0:[], 1:[], 2:[], 3:[], 4:[]}
def hook_stats2(module, input, output): # output = feature map
    mean_spacial_dims = torch.mean(output, dim=(2,3)) # calculating mean over the spacial dimensions
    
    mean_feature_map[(hook_stats2.nr)%5].append(mean_spacial_dims)

    hook_stats2.nr += 1 # hvert 5 kall øker batchnr med 1    




def feature_map_stats2(): 

    all_covariances = []

    for key in mean_feature_map.keys(): # iterating through feature maps
        one_tensor = torch.cat(mean_feature_map[key], dim=0) # creating one tensor per feature map
        one_tensor = one_tensor.detach().cpu().numpy()    
        
        channel_size = one_tensor.shape[1]
        covariance_matrix = np.zeros((channel_size, channel_size))
        e_hat = np.mean(one_tensor, axis=0).reshape(channel_size, 1)  # summing over a row - for b=0 over all channels
      
        for b in range(one_tensor.shape[0]):

            fbc = one_tensor[b,:].reshape(channel_size, 1)
            covariance_matrix += fbc @ np.transpose(fbc) - e_hat @ np.transpose(e_hat)

        covariance_matrix = 1/one_tensor.shape[0] * covariance_matrix
            
        all_covariances.append(covariance_matrix)
    
    return all_covariances
        



def get_eigenvalues(all_covariances, k):

    all_eigenvalues = []


    for cov_matrix in all_covariances:
        eig_vals = np.linalg.eigvals(cov_matrix) # finding all eigenvalues for one covariance matrix
        eig_vals_sorted = eig_vals[eig_vals.argsort()[::-1]]
        eig_vals_sorted = eig_vals_sorted[:k]
        all_eigenvalues.append(eig_vals_sorted)

    return all_eigenvalues

def plot_eigenvalues(all_eigenvalues, modules, title):

    plt.figure(figsize=(10,8))
    
    for eig, i in zip(all_eigenvalues, modules): # eig is a list
        plt.plot(eig, label=f"Eig feature map for module {i}")
   
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Eigenvalue")
    plt.show()



def dataset_eigeinvalues(model, device, dl, title, config):
    count, modules = get_feature_maps(hook_stats2, dl, model, device, config) # this will use the hook_stats2 and append in the mean_feautre map
    all_covariances = feature_map_stats2() # this will give us all the covariance matrices

    k = 15 # nr of eigenvalues
    eigenvalues = get_eigenvalues(all_covariances, k)
    plot_eigenvalues(eigenvalues, modules, title)

    return eigenvalues



def all_datasets(root_dirs, device, dl_our, config):
    t = transformation()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    eig_our = dataset_eigeinvalues(model, device, dl_our["validation"], "Plot of eigenvalues for our dataset", config)

    dl_imagenet_dataset = dataloaders(root_dirs[1], config["batchsize_train"], config["batchsize_test"], t)
    eig_imagenet = dataset_eigeinvalues(model, device, dl_imagenet_dataset["validation"], "Plot of eigenvalues for the imagenet dataset", config)

    train_dataset = datasets.CIFAR10(root=root_dirs[2], train=False, download=True, transform=t)
    d = DataLoader(train_dataset, batch_size=config["batchsize_test"], shuffle=False)
    eig_cifar = dataset_eigeinvalues(model, device, d, "Plot of eigenvalues for the CIFAR10 dataset", config)

    all_eigenvalues = [eig_our, eig_imagenet, eig_cifar]

    return all_eigenvalues




def reproduction_routine(filename_scores, filename_model, dl, device, config):
    """This is the code for the reproduction routine. The saved softmax and the saved model are loaded from
    file. Then the test-function is called with the loaded model to get the softmax scores. Then the saved softmax scores
    and the new softmax scores are compared. But I'm not sure what is meant by comparing the tensors, as in my code it would 
    make sense that the tensors aren't the same so I don't know what is meant by this .. :("""
    
    loaded_predictions = torch.load(filename_scores) # loading the tensor which is saved to file
    
    model = init_model(config) # creating an instance of the default model
    model.load_state_dict(torch.load(filename_model)) # loading the state of our model into this newly created model

    # running the test-function on this model
    softmax_scores, predictions, all_outputs_AP, images, accuracies, aps = test_model(model, dl["test"], device)

    comparison = torch.eq(loaded_predictions, predictions)

    print("Saved predictions:\n")
    print(loaded_predictions)

    print("\nThese predictions:\n")
    print(predictions)

    return comparison



def main():     
        config = {
                "batchsize_train": 32,
                "batchsize_test": 16,
                "epochs": 10,
                "learning_rate": 0.001,
                "nr_classes": 6,
                "model_nr": 0
        }

        root_dir = "mandatory1_data" # ROOT DIRECTORY NEEDS TO BE SET
        t = transformation() # getting the transformation
        dl = dataloaders(root_dir, config["batchsize_train"], config["batchsize_test"], t) # getting the dataloaders for train, test and split
        device = "mps"

        print("\n")

        print("REPORT TASK 1:\n")

        # this trains 3 models with different optimizers and outputs the best based on accuracy
        # this also plots the loss curves for the chosen models, in addition to curves for mean class-wise accuracy
        # this is the saved model!
        model = train_models(root_dir, config, dl, device)

        # saving the model to file
        torch.save(model.state_dict(), 'model.pth')

        # priting out accuracy and average precision for the test set on the final model
        softmax_scores, predictions, all_outputs_AP, images, accuracies, aps = test_model(model, dl["test"], device)
        print("\nFinal mean class-wise accuracy and average precision for the test set:")
        print_stuff(accuracies, aps, 0) 

        # saving the softmax outputs and predictions to file
        torch.save(softmax_scores, "softmax_output_test.pt")
        torch.save(predictions, "predictions_output_test.pt")

        # showing the 10 top and bottom images for each class
        show_ranked_images(model, dl, device)

        print("\n")

        print("REPORT TASK 2:\n")

        # printing out feature map statistics part 1
        result_stats = feature_map_statistics(dl, config, model, device)
        print_feature_map_statistics(result_stats)

        print("\n")

        print("REPORT TASK 3:\n")

        root_dirs = ["mandatory1_data", "ILSVRC2012_img_val", "cifar-10-batches-py"] # ROOT DIRECTORIES NEEDS TO BE SET
        all_datasets(root_dirs, "cpu", dl, config)

        # reproduction routine code
        comparison_tensor = reproduction_routine("predictions_output_test.pt", "model.pth", dl, "cpu", config)

if __name__ == "__main__":
        main()






