for epoch in range(0, 5):
	if epoch > 1:
		x = str(input("Enter 'Yes' to continue training for epoch[{}]: ".format(epoch)))
		if x.lower() == "yes":
			print("training continues.....")
		else:
			break
	
