def mdd(theta,y,x,z):
    e=y-x@theta.reshape(33,2)
    meane=np.mean(e,0)
    e_demean=e-meane
    e_mat=e_demean.dot(e_demean.T)
    z=z.reshape(n,-1)
    z_mat=squareform(pdist(z,metric='euclidean'))
    return (abs(np.mean(e_mat*z_mat)))
def issmaller(a,b,n):
    return (a<=b).all(axis=1).reshape(n,1)
def DL(theta,y,x,z):
    e=y-x@theta.reshape(33,2)
    tot=0
    for l in range(n):
        tot+=np.sum(e*issmaller(z,z[l,:],n),axis=0).reshape(1,2)@np.sum(e*issmaller(z,z[l,:],n),axis=0).reshape(2,1)
    return(tot/n**3)
closing=np.concatenate([sp500.reshape(2275,1),csco.reshape(2275,1),intc.reshape(2275,1)],axis=1)
Y=np.diff(np.log(closing),axis=0)
index=np.array(range(3,2274))
n=len(index)
dim=3
IVlag=3
para_number=30
valid_samplesize=n  
convar=np.concatenate([Y[index-1,:],Y[index-2,:],Y[index-3,:]],axis=1) 
theta_mddW=np.zeros([para_number-3,para_number-3])
theta_mdd=np.zeros([para_number-3,1])
theta_DLW=np.zeros([para_number,para_number])
theta_DL=np.zeros([para_number,1])
x=convar
y=Y[index,:]
for i in range(n):
    xi_demean=x[i,:]-np.mean(x,axis=0)
    for j in range(n):
        xj_demean=(x[j,:]-np.mean(x,axis=0)).reshape(1,9)
        yj_demean=(y[j,:]-np.mean(y,axis=0)).reshape(3,1)
        theta_mddW+=np.kron(xi_demean,np.diag(np.ones(dim))).T@np.kron(xj_demean,np.diag(np.ones(dim)))*np.linalg.norm(convar[i,:]-convar[j,:])/n**2  
        theta_mdd+=np.kron(xi_demean,np.diag(np.ones(dim))).T@yj_demean*np.linalg.norm(convar[i,:]-convar[j,:])/n**2  
theta_mdd=np.linalg.pinv(theta_mddW)@theta_mdd
alpha_mdd=np.mean(y-x@theta_mdd.reshape(9,3),axis=0)
kernelDL=np.zeros([n,n]) 
for i in tqdm(range(n),desc='...'):
    time.sleep(0.01)
    xi=np.concatenate([np.ones(1),x[i,:]])
    for j in range(n):
        xj=np.concatenate([np.ones(1),x[j,:]]).reshape(1,10)
        yj=y[j,:].reshape(3,1)
        for k in range(n):     
            kernelDL[i,j]+=all(convar[k,:]>=convar[i,:])*all(convar[k,:]>=convar[j,:])/n
        theta_DLW+=np.kron(xi,np.diag(np.ones(dim))).T@np.kron(xj,np.diag(np.ones(dim)))*kernelDL[i,j]/n**2  
        theta_DL+=np.kron(xi,np.diag(np.ones(dim))).T@yj*kernelDL[i,j]/n**2  
thetaalpha_DL=np.linalg.pinv(theta_DLW)@theta_DL
alpha_DL=thetaalpha_DL[0:3]
alpha_DL=alpha_DL.flatten()
epsilon_mdd=y-x@(theta_mdd.reshape(9,3))-alpha_mdd
epsilon_DL=y-x@(theta_DL.reshape(9,3))-alpha_DL
sum_mdd=np.zeros([n,dim,para_number-3])
dotHn=np.zeros([n,dim,para_number])
meandoth=np.zeros((dim,para_number-3))
for i in range(n):# 
    latent=-np.kron(x[i,:],np.diag(np.ones(dim)))
    meandoth+=latent/n 
gam_DL=np.zeros([n,n,dim,dim])
for i in range(n):
    for j in range(n):
        doth=-np.kron(x[j,:],np.diag(np.ones(dim)))
        sum_mdd[i,:,:]+=(doth-meandoth)*np.linalg.norm(convar[i,:]-convar[j,:])/n  
        dotHn[i,:,:]+=-np.kron(np.concatenate([np.ones(1),x[j,:]]),np.diag(np.ones(dim)))*(all(convar[j,:]<=convar[i,:]))/n  
        for k in range(n):
            gam_DL[i,j,:,:]+=epsilon_DL[k,:].reshape(dim,1)@epsilon_DL[k,:].reshape(1,dim)*all(convar[k,:]<=convar[i,:])*all(convar[k,:]<=convar[j,:])/n 
Omega=np.zeros([para_number-3,para_number-3])
asd_mdd=np.zeros([para_number-3,para_number-3])
sum_mdd_mean=np.mean(sum_mdd,axis=0) 
for i in range(n):
    doth=-np.kron(x[i,:],np.diag(np.ones(dim)))
    Omega+=(doth-meandoth).T@sum_mdd[i,:,:]/n
    asd_mdd+=(sum_mdd[i,:,:]-sum_mdd_mean).T@epsilon_mdd[i,:].reshape(dim,1)@epsilon_mdd[i,:].reshape(1,dim)@(sum_mdd[i,:,:]-sum_mdd_mean)/n
asd_mdd=np.linalg.pinv(Omega)@asd_mdd@np.linalg.pinv(Omega)
asd_mdda=0   
asd_DL=np.zeros([para_number,para_number])
W_DL=np.zeros([para_number,para_number])
for k in range(n):
    W_DL+=dotHn[k,:,:].T@dotHn[k,:,:]
    for k2 in range(n):
        asd_DL+=dotHn[k,:,:].T@gam_DL[k,k2,:]@dotHn[k2,:,:]
asd_DL=np.linalg.pinv(W_DL)@asd_DL@np.linalg.pinv(W_DL)
print(theta_mdd)
print(theta_DL)
print(np.diag(np.sqrt(asd_mdd/n)))
print(np.diag(np.sqrt(asd_DL/n)))