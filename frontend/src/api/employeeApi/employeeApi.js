export const apiPostFile = async (url = '',file, apartament) => {
	const serverData = {
		ans: { status: null, statusText: '' },
		error: { isError: false, errorText: '' },
		data: [],
	};

    const formData = new FormData();
    formData.append('video', file);
	formData.append('flat', apartament);

	const options = {
		method: 'POST',
		mode: 'cors',
		headers: {
			Authorization: `Bearer ${localStorage.getItem('token')}`,
		},
        body: formData,
	};

	try {
		const answer = await fetch(url, options);

		const { status, statusText } = answer;
		const { ans } = serverData;

		serverData.ans = { ...ans, status, statusText };

		serverData.data = await answer.json();
	} catch (error) {
		serverData.error = { isError: true, errorText: error.message };
	}

	return serverData;
};