// New item form: photos from the camera and from the library add up, with previews and a
// remove button, instead of each pick replacing the last. Without this script the form still
// posts whatever the two pickers hold.
(function () {
  "use strict";
  const form = document.querySelector("[data-photo-form]");
  if (!form) return;
  let chosen;
  try {
    chosen = new DataTransfer();
  } catch (error) {
    return; // no DataTransfer constructor: keep the plain pickers
  }
  const maxPhotos = Number(form.dataset.maxPhotos);
  const maxBytes = Number(form.dataset.maxMb) * 1024 * 1024;
  const previews = form.querySelector("[data-previews]");
  const note = form.querySelector("[data-photo-note]");
  const submit = form.querySelector("[data-submit]");
  const defaultNote = note.textContent;
  const store = document.createElement("input");
  store.type = "file";
  store.name = "photos";
  store.multiple = true;
  store.hidden = true;
  form.appendChild(store);
  let urls = [];

  form.querySelectorAll("input[data-picker]").forEach(function (picker) {
    picker.removeAttribute("name");
    picker.addEventListener("change", function () {
      Array.from(picker.files).forEach(function (file) {
        if (!Array.from(chosen.files).some(function (f) { return same(f, file); })) {
          chosen.items.add(file);
        }
      });
      picker.value = "";
      sync();
    });
  });

  previews.addEventListener("click", function (event) {
    const index = event.target.dataset.remove;
    if (index === undefined) return;
    const next = new DataTransfer();
    Array.from(chosen.files).forEach(function (file, i) {
      if (i !== Number(index)) next.items.add(file);
    });
    chosen = next;
    sync();
  });

  form.addEventListener("submit", function (event) {
    const problem = check();
    if (problem) {
      event.preventDefault();
      note.textContent = problem;
      return;
    }
    submit.disabled = true;
    submit.textContent = "Uploading...";
  });

  function same(a, b) {
    return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
  }

  function check() {
    const files = Array.from(chosen.files);
    if (!files.length) return "Add at least one photo.";
    if (files.length > maxPhotos) return "Up to " + maxPhotos + " photos. Remove some.";
    const big = files.find(function (file) { return file.size > maxBytes; });
    return big ? big.name + " is too big." : "";
  }

  function sync() {
    store.files = chosen.files;
    urls.forEach(URL.revokeObjectURL);
    urls = [];
    previews.replaceChildren();
    Array.from(chosen.files).forEach(function (file, i) {
      const url = URL.createObjectURL(file);
      urls.push(url);
      const item = document.createElement("li");
      const image = document.createElement("img");
      image.src = url;
      image.alt = "Photo " + (i + 1);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.dataset.remove = String(i);
      item.append(image, remove);
      previews.append(item);
    });
    const count = chosen.files.length;
    const ready = count === 1 ? "1 photo ready." : count + " photos ready.";
    note.textContent = count ? check() || ready : defaultNote;
  }
})();

// A server error (5xx) is not swapped in, so without this a request that failed would look like
// nothing happened: the drafting note would pulse forever, Confirm would do nothing.
document.addEventListener("htmx:responseError", function (event) {
  "use strict";
  const toast = document.getElementById("toast");
  if (!toast) return;
  const status = event.detail && event.detail.xhr ? event.detail.xhr.status : 0;
  const box = document.createElement("div");
  box.className = "flash error";
  box.setAttribute("role", "alert");
  const text = document.createElement("p");
  text.textContent = "Something went wrong" + (status ? " (" + status + ")" : "") +
    ". Reload the page and try again.";
  box.append(text);
  toast.replaceChildren(box);
  document.querySelectorAll(".drafting .pulse").forEach(function (note) {
    note.classList.remove("pulse");
    note.textContent = "The draft did not finish. Reload the page to try again.";
  });
});
