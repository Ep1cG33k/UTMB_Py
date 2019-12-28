macro "find Anomaly" {
	setOption("ExpandableArrays", true);
	// ask for a file to be imported
	fileName = File.openDialog("Select the file to import");
	allText = File.openAsString(fileName);

	Dialog.create("OCT or CSV as Input to AD Model")
	items = newArray("OCT", "CSV");
  	Dialog.addRadioButtonGroup("What file type did you input to AD model in PyCharm?", items, 1, 2, "OCT");
	Dialog.show();
	type = Dialog.getRadioButton;
	//Split the CSV file into an array of lines 
	lines = split(allText, "\n"); 
	nindents = lengthOf(lines); 
	slices = newArray;
	strips = newArray;
	layer = newArray;
	y = newArray;
	x = newArray;
	for (i = 0; i < nindents; i++) {
		data = split(lines[i], "\,"); 
		slices[i] = data[0];
		strips[i] = data[1];
		n = data[1];
		layer[i] = data[2];
		if (type=="OCT"){
			y[i] = data[4];
			x[i] = data[5];
		}
	}

	default = 1;
	number = 1;
	while(true){
		default = number;
		Dialog.create("View Anomaly");
		Dialog.addSlider('Anomaly #', 1, nindents, default);
		//print(number);
		Dialog.show();
		number = Dialog.getNumber();
		viewAnomalyStrip(slices[number-1], strips[number-1]);
		if(type=="OCT"){
			viewAnomalyROI(y[number-1], x[number-1], strips[number-1], 20);
		}
		print(layer[number-1]);
	}
}

function viewAnomalyStrip(sliceNum, stripNum){
	setSlice(sliceNum + 1);
	makeRectangle(64*(stripNum), 0, 64, 256);
}

function viewAnomalyROI(y, x, n, range){
	//width of oval
	n = (parseInt(n)) * 64;
	x = parseInt(x);
	y = parseInt(y);
	makeOval((n+x)-(range/2), y-(range/2), range, range);
	//makeRectangle(y-(range/2), x-(range/2), range, range);
}
