const btn = document.querySelector('button')
const inputs = document.querySelector('form')
btn.addEventListener('click', () => {

    const name = (inputs.elements["name"].value).trim()
    const email = (inputs.elements["email"].value).trim()
    const msg = (inputs.elements["message"].value.trim())
    const phone = (inputs.elements["phone"].value).trim()
    if (!name.length > 0 || !email.length > 0 || !msg.length > 0 || !phone.length > 0) {
        alert("All fields are mandatory")
        return
    }
    Email.send({
        Host: "smtp.elasticemail.com",
        Username: "ale.ava0208@gmail.com",
        Password: "DB6E065A57351D6B5052F0C612A92D921FB2",
        To: "ale.ava0208@gmail.com",
        From: email,
        Subject: "Contact Us Query By the Customer",
        Body: msg + "<br>" + name + "<br>" + phone
    }).then(msg => alert(msg))
})
