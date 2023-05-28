export const createJC = async (url = '', id, name, coords, adress, date) => {
    const options = {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Accept: 'application/json',
			// Authorization: `Token ${token}`,
		},
		body: JSON.stringify({ id, name, coords, adress, date }),
	};

    const answer = await fetch(url, options);

    return answer;
}

export const createFrame = async (url = '', id, name, countEntries, countApartaments, countFloors) => {
    const options = {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Accept: 'application/json',
			// Authorization: `Token ${token}`,
		},
		body: JSON.stringify({ id, name, countEntries, countApartaments, countFloors }),
	};

    const answer = await fetch(url, options);

    return answer;
}

export const apiCreateProject = async (url = '', parameters = {}) => {
	console.log(url);
	const serverData = {
		ans: { status: null, statusText: '' },
		error: { isError: false, errorText: '' },
		data: [],
	};

	const options = {
		method: 'POST',
		mode: 'cors',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${localStorage.getItem('token')}`,
		},
		body: JSON.stringify({parameters})
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