import torch

def targeted_adversarial_attack(model, x, y_target, epochs=1000, l=15000, eps=0.03):
    
    x_adv, x = x.cuda(), x.cuda()
    min_loss = float('inf')
    x_prime, y_target = x.cuda(), torch.LongTensor([y_target]).cuda()
    x_prime.requires_grad_(True)
    
    model.to(model.device)
    model.eval()
    
    for i in range(epochs):
        with torch.set_grad_enabled(True):
          
            output = model(x_prime)

            y_predict = torch.max(output, 1)[1]
            
            # get P(y_target | x')
            prob = torch.nn.functional.softmax(output, dim = 1)[0][y_target]
            loss = torch.nn.CrossEntropyLoss()(output, y_target)
            loss += (l / 2) * ((x_prime - x).norm(p=float("inf")))
            
            # minimize cross entropy loss of fraud classification
            if loss < min_loss and y_predict == y_target.data:
                x_adv = x_prime
                min_loss = loss
                #print("New Prob {}".format(torch.nn.functional.softmax(vgg16(x_adv),dim = 1)[0][812].data))

            # get P(y_target | x') w.r.t. x'
            prob.backward()
            
            # update x' by the gradient of prob(y_target | x') w.r.t. x'
            tmp = (x_prime + (l * x_prime.grad)).cuda()

            # clip x' value within (x - eps, x + eps)
            tmp = torch.where(tmp > x + eps, x + eps, tmp)
            tmp = torch.where(tmp < x - eps, x - eps, tmp)
            x_prime = torch.autograd.Variable(tmp, requires_grad=True).cuda()
            
    return x_adv

def non_targeted_adversarial_attack(model, x, epochs=5, l=15000, eps=0.03):
    
    x_adv, x = x.cuda(), x.cuda()
    max_loss = float('-inf')
    x_prime = x.cuda()
    x_prime.requires_grad_(True)
    
    model.to(model.device)
    model.eval()
    
    # get original prediction and confidence
    y = torch.max(model(x), 1)[1]
    prob = torch.nn.functional.softmax(model(x), dim = 1)[0][y]
    #print("Original Predict class: {}, Probability: {}".format(classes[int(y[0].data)], prob[0].data))
    
    for i in range(epochs):
        with torch.set_grad_enabled(True):
          
            output = model(x_prime)
            prob = torch.nn.functional.softmax(output, dim = 1)[0][y]
            loss = torch.nn.CrossEntropyLoss()(output, y)

            if loss > max_loss:
                x_adv = x_prime
                max_loss = loss
                #print("New Prob {}".format(torch.nn.functional.softmax(vgg16(x_adv),dim = 1)[0][y][0].data))
            
            # get P(y | x') w.r.t. x'
            prob.backward()

            # update x' by the ascending gradient of prob
            tmp = (x_prime - (l * x_prime.grad)).cuda()
            # clip x' value within (x - eps, x + eps)
            tmp = torch.where(tmp > x + eps, x + eps, tmp)
            tmp = torch.where(tmp < x - eps, x - eps, tmp)
            x_prime = torch.autograd.Variable(tmp, requires_grad=True).cuda()
    
    # output = vgg16(x_adv)
    # value, category = torch.max(output, 1)
    # prob = torch.nn.functional.softmax(output, dim = 1)[0][category]
    # print("New Predict class: {}, Probability: {}".format(classes[int(category[0].data)], prob[0].data))
    return x_adv