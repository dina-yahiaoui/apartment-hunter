// Apartment Hunter : conversions sqft -> m² en direct et carte Leaflet de l'emplacement

const SQFT_EN_M2 = 0.09290304;

// --- Équivalent en m² affiché sous chaque champ de surface ---
function majM2(champ) {
  const cible = document.querySelector(`[data-m2-pour="${champ.id}"]`);
  if (!cible) return;
  const sqft = Number(champ.value);
  cible.textContent = champ.value === "" || Number.isNaN(sqft)
    ? "≈ — m²"
    : `≈ ${Math.round(sqft * SQFT_EN_M2).toLocaleString("fr-FR")} m²`;
}

document.querySelectorAll("[data-m2]").forEach((champ) => {
  majM2(champ);
  champ.addEventListener("input", () => majM2(champ));
});

// --- Carte : clic = position choisie ; changement de code postal = recentrage sur sa médiane ---
const conteneurCarte = document.getElementById("carte");

if (conteneurCarte && window.L) {
  const champLat = document.getElementById("lat");
  const champLong = document.getElementById("long");
  const champChoisie = document.getElementById("position_choisie");
  const statut = document.getElementById("statut-carte");
  const selectZip = document.getElementById("zipcode");
  const medianesZip = JSON.parse(document.getElementById("donnees-zip").textContent);

  const depart = [Number(conteneurCarte.dataset.lat), Number(conteneurCarte.dataset.long)];
  const carte = L.map(conteneurCarte, { scrollWheelZoom: false }).setView(depart, 13);

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">contributeurs OpenStreetMap</a>',
  }).addTo(carte);

  const marqueur = L.marker(depart, { draggable: true, keyboard: true, title: "Emplacement de la maison" }).addTo(carte);

  function placer(latlng, choisie) {
    marqueur.setLatLng(latlng);
    champLat.value = latlng.lat.toFixed(5);
    champLong.value = latlng.lng.toFixed(5);
    champChoisie.value = choisie ? "1" : "0";
    statut.textContent = choisie
      ? `Point choisi sur la carte (${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}).`
      : "Position par défaut : centre du code postal. Cliquez sur la carte pour placer la maison précisément.";
  }

  carte.on("click", (e) => placer(e.latlng, true));
  marqueur.on("dragend", () => placer(marqueur.getLatLng(), true));

  selectZip.addEventListener("change", () => {
    const centre = medianesZip[selectZip.value];
    if (!centre) return;
    const latlng = L.latLng(centre[0], centre[1]);
    placer(latlng, false);
    carte.setView(latlng, 13);
  });
}
