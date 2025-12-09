//  MENU 

// seleciona o botao de menu e o nav
const menuToggle = document.getElementById('menuToggle');
const navMenu = document.getElementById('navMenu');

menuToggle.addEventListener('click', function () {
    // alterna a classe open para mostrar/ocultar o menu no mobile, achei bonitinho
    navMenu.classList.toggle('open');
});
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-link');
// funçao que verifica qual seçao está visivel e marca o link respectivo
function marcarLinkAtivo() {
    let indexAtivo = sections.length;

    // percorre as seçoes de baixo pra cima
    while (--indexAtivo && window.scrollY + 100 < sections[indexAtivo].offsetTop) {}

    navLinks.forEach(link => link.classList.remove('active'));
    navLinks[indexAtivo].classList.add('active');
}
// chama a função no scroll
window.addEventListener('scroll', marcarLinkAtivo);
//  SCROLL SUAVE  
navLinks.forEach(link => {
    link.addEventListener('click', function (event) {
        // impede o comportamento padrão do link (pular direto)
        event.preventDefault();

        // fecha o menu mobile ao clicar em um link 
        navMenu.classList.remove('open');

        // seleciona a seção alvo pelo ID do href
        const idAlvo = this.getAttribute('href');
        const secaoAlvo = document.querySelector(idAlvo);

        // faz o scroll suave até a seçao
        window.scrollTo({
            top: secaoAlvo.offsetTop - 60, // compensa a altura do header fixo
            behavior: 'smooth'
        });
    });
});

// FILTRO DOS PROJETOS 
const botoesFiltro = document.querySelectorAll('.btn-filtro');
const cardsProjetos = document.querySelectorAll('.projeto-card');
botoesFiltro.forEach(botao => {
    botao.addEventListener('click', function () {
        // remove a classe active dos outros botoes
        botoesFiltro.forEach(btn => btn.classList.remove('active'));
        // adiciona active no botao clicado
        this.classList.add('active');

        const filtro = this.getAttribute('data-filter');

        cardsProjetos.forEach(card => {
            const categoria = card.getAttribute('data-category');

            // se o filtro for todos mostra tudo, se ñ filtra pela categoria
            if (filtro === 'todos' || categoria === filtro) {
                card.classList.remove('hidden');
            } else {
                card.classList.add('hidden');
            }
        });
    });
});

//  VALIDAÇAO DO FORMULARIO DE CONTATO  
const formContato = document.getElementById('contatoForm');
const mensagemRetorno = document.getElementById('mensagemRetorno');
// funçao simples para validar e-mail
function emailValido(email) {
    // expressao regular simples para validar formato de email
    const padrao = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return padrao.test(email);
}

formContato.addEventListener('submit', function (event) {
    event.preventDefault(); // impede o envio real do formulario

    // pega os campos
    const campoNome = document.getElementById('nome');
    const campoEmail = document.getElementById('email');
    const campoMensagem = document.getElementById('mensagem');

    // assume que está tudo válido no começo
    let formularioValido = true;

    // limpa mensagens e estados anteriores
    mensagemRetorno.textContent = '';
    document.querySelectorAll('.grupo-input').forEach(grupo => {
        grupo.classList.remove('invalido');
        const spanErro = grupo.querySelector('.erro-msg');
        if (spanErro) {
            spanErro.textContent = '';
        }
    });
    // validaçao do nome
    if (campoNome.value.trim().length < 3) {
        const grupo = campoNome.closest('.grupo-input');
        grupo.classList.add('invalido');
        grupo.querySelector('.erro-msg').textContent = 'Informe um nome com pelo menos 3 caracteres.';
        formularioValido = false;
    }

    // validaçao do e-mail
    if (!emailValido(campoEmail.value.trim())) {
        const grupo = campoEmail.closest('.grupo-input');
        grupo.classList.add('invalido');
        grupo.querySelector('.erro-msg').textContent = 'Informe um e-mail válido.';
        formularioValido = false;
    }

    // validaçao da mensagem
    if (campoMensagem.value.trim().length < 10) {
        const grupo = campoMensagem.closest('.grupo-input');
        grupo.classList.add('invalido');
        grupo.querySelector('.erro-msg').textContent = 'A mensagem deve ter pelo menos 10 caracteres.';
        formularioValido = false;
    }

    // se algum campo estiver invalido, mostra feedback genérico
    if (!formularioValido) {
        mensagemRetorno.style.color = '#f97373';
        mensagemRetorno.textContent = 'Por favor, corrija os campos destacados.';
        return;
    }

    // se chegou aqui, consideramos o formulário válido
    mensagemRetorno.style.color = '#4ade80';
    mensagemRetorno.textContent = 'Mensagem enviada com sucesso!';

    // aqui você poderia integrar com um serviço de envio de email,
    // mas para o trabalho atual, a simulação e suficiente.
    formContato.reset();
});

// ANO ATUAL NO RODAPE

// preenche o span com o ano atual automaticamente
const anoSpan = document.getElementById('anoAtual');
if (anoSpan) {
    anoSpan.textContent = new Date().getFullYear();
}

//  alternar tema claro/escuro 
const body = document.body;
const themeToggleBtn = document.getElementById('themeToggle');

// recuperar preferência salva 
const storedTheme = localStorage.getItem('theme');

if (storedTheme === 'dark') {
    body.classList.add('dark-theme');
    if (themeToggleBtn) {
        themeToggleBtn.textContent = '☀️';
        themeToggleBtn.setAttribute('aria-label', 'Alternar tema para claro');
    }
} else {
    if (themeToggleBtn) {
        themeToggleBtn.textContent = '🌙';
        themeToggleBtn.setAttribute('aria-label', 'Alternar tema para escuro');
    }
}

if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        const isDark = body.classList.toggle('dark-theme');

        if (isDark) {
            localStorage.setItem('theme', 'dark');
            themeToggleBtn.textContent = '☀️';
            themeToggleBtn.setAttribute('aria-label', 'Alternar tema para claro');
        } else {
            localStorage.setItem('theme', 'light');
            themeToggleBtn.textContent = '🌙';
            themeToggleBtn.setAttribute('aria-label', 'Alternar tema para escuro');
        }
    });
}
