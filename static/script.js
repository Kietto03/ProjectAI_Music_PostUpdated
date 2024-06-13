document.getElementById("file-upload").addEventListener("change", function () {
  const fileName = this.files[0].name;
  document.getElementById("file-name").textContent = `Selected file: ${fileName}`;
});

document.getElementById("uploadForm").addEventListener("submit", function () {
  document.getElementById("loadingSpinner").style.display = "block";
});
