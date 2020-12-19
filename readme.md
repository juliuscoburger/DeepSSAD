# Weak supervised explainable deep anomaly detection



* 3 Losses 
* different datasets and OE methods




## How to run?
`model = DeepAD("DSVDD", net_name, d_name=dataset_name, epochs=100, lr=1e-4, nc=c, lr_milestones=[70], loss=loss, 
                do_print=do_print, do_pretrain=do_pretrain, pretrain_epochs=20, noise=noise, weak_supervision=weak_supervision, 
                weak_supervision_size=weak_supervision_size, device="cpu", weak_supervision_set=weak_supervision_set, out_file=out_file)
                
model.train()
model.test()`





