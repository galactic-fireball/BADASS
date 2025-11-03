
function addTabListeners() {
	const tabs = document.querySelectorAll('.tab');
	const contents = document.querySelectorAll('.tab-content');

	tabs.forEach(tab => {
		tab.addEventListener('click', () => {
			console.log('CLICK');
			tabs.forEach(t => t.classList.remove('active'));
			contents.forEach(c => c.classList.remove('active'));
			contents.forEach(c => console.log(c));

			tab.classList.add('active');
			tabContent = document.getElementById(tab.dataset.content);
			console.log(tabContent);
			tabContent.classList.add('active');
		});
	});
}

addTabListeners();
