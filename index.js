

const url = 'http://localhost:8000/';

/*let data = {
   title: title,
   blob: blob
} */

var req = new Request(url, {
    method: 'GET',
    //body: data,
    headers: new Headers()
});
/*
function sendenfunktion(){

  fetch(url)
  .then((resp) => resp.json())
  .then(function(data) {
  let authors = data.results;
  element_ergebnisfeld.innerHTML=data.Hello;
})
  .catch(function(error) {
    console.log(error);
  });

}
*/
