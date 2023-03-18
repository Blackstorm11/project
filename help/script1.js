const imageUpload = document.getElementById('imageUpload')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
]).then(start)

async function start(){
  const container=document.createElement('div')
  container.style.position='relative'
  document.body.append(container)
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  
    document.body.append('Loaded')
    imageUpload.addEventListener('change',async()=>{
      const image = await faceapi.bufferToImage(imageUpload.files[0])
      container.append(image)
      canvas = faceapi.createCanvasFromMedia(image)
      container.append(canvas)
        const displaySize={width: image.width, height:image.height}
        faceapi.matchDimensions(canvas, displaySize)

        
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections=faceapi.resizeResults(detections,displaySize)
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
      
        results.forEach((result, i) => {
          const box = resizedDetections[i].detection.box
          const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
          drawBox.draw(canvas)
        });
    })
}



// function loadLabeledImages(){
  
//   const labels=['omkar','lachlan','hrithik','prajakta']
//   return Promise.all(
//     labels.map(async label=>{
     
//       const descriptions=[]
//       for (let i=1;i<=2; i++){
//         const img= await faceapi.fetchImage(`https://github.com/Blackstorm11/project/tree/main/face_Recognition/${label}/${i}.jpg`)
//         const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
//         descriptions.push(detections.descriptor)
//       }

//       return new faceapi.LabeledFaceDescriptor(label,descriptions)
//     })
//   )
// }


async function loadLabeledImages() {
  const labels = ['omkar','hrithik_kantak','omkar_redkar','prajakta_kolambkar'];
  // const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
  const labeledDescriptors = [];

  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    const descriptors = [];

    for (let j = 1; j <= 2; j++) {
      const img = await faceapi.fetchImage(`./labeled_images/${label}/${j}.jpg`);
      const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
      descriptors.push(detection.descriptor);
    }

    labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
  }

  return labeledDescriptors;
}




