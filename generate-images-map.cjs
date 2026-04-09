const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, 'public', 'images');
const destFile = path.join(__dirname, 'public', 'data', 'images.json');

try {
  const files = fs.readdirSync(srcDir);
  const imageMap = {};

  files.forEach(file => {
    if (file.match(/\.(jpg|jpeg|png)$/i)) {
      const rollNo = path.parse(file).name;
      imageMap[rollNo] = file;
    }
  });

  fs.writeFileSync(destFile, JSON.stringify(imageMap, null, 2));
  console.log(`✅ Image manifest generated: ${Object.keys(imageMap).length} images mapped.`);
} catch (err) {
  console.error("Error generating image manifest:", err);
}
