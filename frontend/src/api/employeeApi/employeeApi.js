export const apiPostFile = async (url = '',file, id, analysis, date, is_analysed, flat) => {
	const serverData = {
		ans: { status: null, statusText: '' },
		error: { isError: false, errorText: '' },
		data: [],
	};
	console.log(file);
    const formData = new FormData();
    formData.append('video', file);

	const data = {
		id,
        video: formData,
        analysis,
        date,
        is_analysed,
        flat
	}

	console.log(data);

	const options = {
		method: 'POST',
		mode: 'cors',
		headers: {
			'Content-Type': 'multipart/form-data',
			Authorization: `Bearer ${localStorage.getItem('token')}`,
		},
        body: JSON.stringify( data ),
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